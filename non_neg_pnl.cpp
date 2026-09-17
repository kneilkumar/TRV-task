#include <nats.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static std::ofstream PnlFile;
static std::ofstream DiscoveryLog;
static std::ofstream MDLogFile;

static const std::chrono::seconds kWarmupPeriod{120};
static const long long kEntryOffsetTicks = 1;
static const int kMaxExitRungs = 4;
static const double kTimeoutHalfLives = 3.0;
static const double kMoverDominanceShare = 0.5;
static const double kZScoreEntryThreshold = 1.5;
static const long long kEntrySize = 4;
static const int kOrderTimeoutMs = 2000;
static const size_t kMaxHistPoints = 20000;

enum class Role { UNKNOWN, DRIVER, PIGGYBACK };
enum class Phase { WARMUP, LIVE };

static Phase gPhase = Phase::WARMUP;

struct InstrumentMeta {
    long long ticksize = 1;
    long long ref_price = 0;
    long long last_traded_price = 0;
    long long min_vol = 1;
    long long max_vol = 1;
    long long max_tps = 0;
    long long pos_lim = 0;
    long long band = 0;
};

struct ExitRung {
    std::string id;
    long long price = 0;
    long long size = 0;
    long long filled = 0;
    bool posted = false;
};

// Async order-entry pipeline: at most ONE request (add or cancel) in flight
// per instrument at a time. Anything else that wants to fire while one is
// pending gets queued and issued automatically the moment the current one's
// ack arrives. This trades raw speed for a strictly serialized, easy-to-
// reason-about-and-debug order flow: exactly one thing happening at a time,
// no overlapping in-flight requests to untangle when something goes wrong.
enum class ReqKind { NONE, ENTRY_ADD, EXIT_ADD, ENTRY_CANCEL, EXIT_CANCEL };

struct QueuedAction {
    ReqKind kind = ReqKind::NONE;
    char side = 0;            // for *_ADD
    long long price = 0;      // for *_ADD
    long long size = 0;       // for *_ADD
    std::string cancelId;     // for *_CANCEL: the resting order id to cancel
    int exitRungIndex = -1;   // for EXIT_ADD/EXIT_CANCEL: which rung this is
};

struct InFlightRequest {
    ReqKind kind = ReqKind::NONE;
    std::string orderId;
    char side = 0;
    long long price = 0;
    long long size = 0;
    int exitRungIndex = -1;
    long long sentNs = 0;
    std::string reqId;
};

struct InstrumentState {
    std::string feed;
    InstrumentMeta meta;
    Role role = Role::UNKNOWN;
    int cluster = -1;

    long long bidPx = -1, askPx = -1, bidVol = -1, askVol = -1;
    double fv = 0.0;

    std::deque<std::pair<long long, double>> midHist;
    std::vector<double> warmupMids;

    double trueMean = 0.0;
    double stdev = 0.0;
    double halfLife = 0.0;
    double zScore = 0.0;

    std::string moverSender;
    std::unordered_map<std::string, long long> aggressorVolume;
    long long totalDriverVolume = 0;

    long long lastTarget = -1;
    char moverSide = 0;

    bool entryActive = false;
    std::string entryId;
    char entrySide = 0;
    long long entryPrice = -1;
    long long entryFilled = 0;
    long long entryTsNs = 0;

    bool haveExit = false;
    std::vector<ExitRung> exitLadder;

    // Async request pipeline state (see ReqKind/InFlightRequest above).
    bool awaitingAck = false;
    InFlightRequest inFlight;
    std::deque<QueuedAction> actionQueue;

    long long position = 0;
    double realizedPnl = 0.0;
    long long orderSeq = 0;
};

static std::string gSender;
static long long gReqSeq = 0;

static std::string genReqId() {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "R%07lld", ++gReqSeq % 10000000LL);
    return std::string(buf);
}
static natsConnection* gConn = nullptr;
static std::map<std::string, InstrumentState> gInstruments;

struct RawMsg {
    std::string subject;
    std::string payload;
    long long recvNs = 0;
};

static std::deque<RawMsg> gQueue;
static std::mutex gMutex;
static std::condition_variable gCv;

static std::atomic<bool> gShuttingDown{false};

static void flushAllLogs() {
    if (PnlFile.is_open()) PnlFile.flush();
    if (DiscoveryLog.is_open()) DiscoveryLog.flush();
    if (MDLogFile.is_open()) MDLogFile.flush();
}

static void onSignal(int) {
    gShuttingDown.store(true);
    gCv.notify_all();
}

static long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

static const char* envOr(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? v : fallback;
}

static void openMDLog() {
    MDLogFile.open(envOr("MD_LOGGING", "/data/md.csv"));
    if (MDLogFile.is_open())
        MDLogFile << "recv_ns,wire_ts,feed,subject,sender,type,id,vol,price,incoming,resting,matchid,aggressor_side\n";
}

static InstrumentMeta parseMeta(const std::string& value) {
    InstrumentMeta m;
    for (auto& kv : split(value, ' ')) {
        auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        std::string k = kv.substr(0, eq);
        long long v = std::atoll(kv.substr(eq + 1).c_str());
        if (k == "ticksize") m.ticksize = v;
        else if (k == "ref_price") m.ref_price = v;
        else if (k == "last_traded_price") m.last_traded_price = v;
        else if (k == "min_volume") m.min_vol = v;
        else if (k == "max_volume") m.max_vol = v;
        else if (k == "max_tps") m.max_tps = v;
        else if (k == "position_limit") m.pos_lim = v;
        else if (k == "band") m.band = v;
    }
    return m;
}

static std::pair<std::string, std::string> splitId(const std::string& id) {
    auto p = id.find(':');
    if (p == std::string::npos) return {id, ""};
    return {id.substr(0, p), id.substr(p + 1)};
}

static double fairValue(long long bidPx, long long bidVol, long long askPx, long long askVol,
                         long long ticksize, long long refPrice) {
    if (bidPx < 0 && askPx < 0) return static_cast<double>(refPrice);
    if (bidPx < 0) return static_cast<double>(askPx);
    if (askPx < 0) return static_cast<double>(bidPx);
    long long tick = ticksize > 0 ? ticksize : 1;
    if ((askPx - bidPx) >= 2 * tick || bidVol <= 0 || askVol <= 0)
        return (static_cast<double>(bidPx) + static_cast<double>(askPx)) / 2.0;
    return (static_cast<double>(bidPx) * askVol + static_cast<double>(askPx) * bidVol) /
           static_cast<double>(bidVol + askVol);
}

static double meanOf(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double computeHalfLife(const std::vector<double>& mids) {
    if (mids.size() < 3) return 0.0;
    std::vector<double> x(mids.begin(), mids.end() - 1);
    std::vector<double> y(mids.begin() + 1, mids.end());
    double mx = meanOf(x), my = meanOf(y);
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
    }
    if (sxx <= 0) return 0.0;
    double phi = sxy / sxx;
    if (phi <= 0.0 || phi >= 1.0) return 0.0;
    return -std::log(2.0) / std::log(phi);
}

static double computeStdev(const std::vector<double>& mids, double mean) {
    if (mids.size() < 2) return 0.0;
    double ss = 0.0;
    for (double v : mids) ss += (v - mean) * (v - mean);
    return std::sqrt(ss / static_cast<double>(mids.size() - 1));
}

static double computeZScore(double fv, double trueMean, double stdev) {
    if (stdev < 1e-9) return 0.0;
    return (fv - trueMean) / stdev;
}

static long long clampToBand(long long price, const InstrumentMeta& meta) {
    long long lo = meta.ref_price - meta.band;
    long long hi = meta.ref_price + meta.band;
    long long tick = meta.ticksize > 0 ? meta.ticksize : 1;
    price = std::max(lo, std::min(hi, price));
    return (price / tick) * tick;
}

static std::string genOrderId(char prefix, InstrumentState& st) {
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%c%07lld", prefix, ++st.orderSeq % 10000000LL);
    return std::string(buf);
}

// Raw async publish: fires the request and returns immediately. No waiting
// for a reply here — replies come back on our own inbox subject and get
// routed through the normal gQueue/dispatch path just like bbo/md messages,
// so the single consumer thread is never blocked waiting on the exchange.
//
// replySubject encodes gSender/feed/reqId so handleOrderReply can route the
// reply straight back to the right instrument without a separate lookup
// table. NOTE: verify natsConnection_PublishRequestString against whatever
// nats.c actually exposes in this build — this is the "publish with a
// reply-to subject, don't wait" primitive; if the linked client names it
// differently, swap the call, the logic around it is unaffected.
static void publishRequestAsync(const std::string& feed, const std::string& reqId, const std::string& body) {
    std::string subj = "ex.req." + gSender;
    std::string replySubj = "INBOX." + gSender + "." + feed + "." + reqId;
    natsStatus s = natsConnection_PublishRequestString(gConn, subj.c_str(), replySubj.c_str(), body.c_str());
    if (s != NATS_OK) {
        std::cout << "publish failed [" << body << "]: " << natsStatus_GetText(s) << std::endl;
    }
}

static void issueNextQueuedAction(InstrumentState& st);

// requestAdd/requestCancel are the only entry points that touch the
// exchange. If st.awaitingAck is already true, the action is queued instead
// of sent — it will fire automatically, in order, once the current in-flight
// request's ack comes back (see handleOrderReply -> issueNextQueuedAction).
// This guarantees at most one outstanding request per instrument at any
// time, by construction, rather than by convention.
static void requestAdd(InstrumentState& st, char side, long long price, long long size,
                        ReqKind kind, int exitRungIndex = -1) {
    if (st.awaitingAck) {
        QueuedAction a;
        a.kind = kind;
        a.side = side;
        a.price = price;
        a.size = size;
        a.exitRungIndex = exitRungIndex;
        st.actionQueue.push_back(a);
        return;
    }
    std::string id = genOrderId(side == 'B' ? 'B' : 'A', st);

    if (kind == ReqKind::ENTRY_ADD) {
        st.entryActive = true;
        st.entryId = id;
        st.entrySide = side;
        st.entryPrice = price;
        st.entryFilled = 0;
        st.entryTsNs = nowNs();
        DiscoveryLog << nowNs() << ",entry_armed," << st.feed << ",side=" << side << " px=" << price << "\n";
    } else if (kind == ReqKind::EXIT_ADD) {
        ExitRung rung;
        rung.id = id;
        rung.price = price;
        rung.size = size;
        rung.filled = 0;
        rung.posted = true;
        st.exitLadder.push_back(rung);
    }

    std::ostringstream body;
    body << gSender << " A " << st.feed << " " << id << " " << side << " " << size << " " << price << " L";
    st.awaitingAck = true;
    st.inFlight = InFlightRequest{};
    st.inFlight.kind = kind;
    st.inFlight.orderId = id;
    st.inFlight.side = side;
    st.inFlight.price = price;
    st.inFlight.size = size;
    st.inFlight.exitRungIndex = exitRungIndex;
    st.inFlight.sentNs = nowNs();
    st.inFlight.reqId = id;
    std::cout << "[ORDER] " << body.str() << std::endl;
    publishRequestAsync(st.feed, id, body.str());
}

static void requestCancel(InstrumentState& st, const std::string& cancelId, ReqKind kind,
                           int exitRungIndex = -1) {
    if (st.awaitingAck) {
        QueuedAction a;
        a.kind = kind;
        a.cancelId = cancelId;
        a.exitRungIndex = exitRungIndex;
        st.actionQueue.push_back(a);
        return;
    }
    std::string reqId = genReqId();
    std::ostringstream body;
    body << gSender << " C " << st.feed << " " << cancelId;
    st.awaitingAck = true;
    st.inFlight = InFlightRequest{};
    st.inFlight.kind = kind;
    st.inFlight.orderId = cancelId;
    st.inFlight.exitRungIndex = exitRungIndex;
    st.inFlight.sentNs = nowNs();
    st.inFlight.reqId = reqId;
    std::cout << "[CANCEL] " << body.str() << std::endl;
    publishRequestAsync(st.feed, reqId, body.str());
}

// Called once the current in-flight request's ack has been fully processed.
// Fires the next queued action for this instrument, if any — this is what
// turns "post 4 exit rungs" or "cancel 4 remaining rungs" from 4 sequential
// blocking calls into 4 sequential async ones, one per ack, with the same
// end state but never freezing the consumer thread.
static void issueNextQueuedAction(InstrumentState& st) {
    if (st.actionQueue.empty()) return;
    QueuedAction a = st.actionQueue.front();
    st.actionQueue.pop_front();
    if (a.kind == ReqKind::ENTRY_ADD || a.kind == ReqKind::EXIT_ADD) {
        requestAdd(st, a.side, a.price, a.size, a.kind, a.exitRungIndex);
    } else if (a.kind == ReqKind::ENTRY_CANCEL || a.kind == ReqKind::EXIT_CANCEL) {
        requestCancel(st, a.cancelId, a.kind, a.exitRungIndex);
    }
}

static void postExitLadder(InstrumentState& st) {
    if (st.haveExit) return;
    long long span = std::llabs(st.entryPrice - static_cast<long long>(std::llround(st.trueMean)));
    int rungs = 2;
    if (std::fabs(st.zScore) > 2.0) rungs = 3;
    if (std::fabs(st.zScore) > 3.0) rungs = kMaxExitRungs;
    rungs = std::max(1, rungs);

    char exitSide = (st.entrySide == 'S') ? 'B' : 'S';
    long long remaining = st.entryFilled;
    if (remaining <= 0) return;

    st.exitLadder.clear();
    // Mark haveExit true up front so we don't re-enter this while rungs are
    // still being issued one-by-one across multiple acks.
    st.haveExit = true;

    for (int i = 1; i <= rungs && remaining > 0; ++i) {
        long long offset = span * i / (rungs + 1);
        long long px = (exitSide == 'B') ? st.entryPrice - offset : st.entryPrice + offset;
        px = clampToBand(px, st.meta);
        long long sz = (i == rungs) ? remaining : std::max<long long>(1, st.entryFilled / rungs);
        sz = std::min(sz, remaining);
        // requestAdd sends now if nothing's in flight, or queues behind
        // whatever is — either way rung i-1 will actually hit the wire only
        // after rung i-2's ack (if any) has come back. exitRungIndex lets
        // handleOrderReply know which rung this ack belongs to.
        requestAdd(st, exitSide, px, sz, ReqKind::EXIT_ADD, i - 1);
        remaining -= sz;
    }
    DiscoveryLog << nowNs() << ",exit_ladder_planned," << st.feed << ",rungs=" << rungs
                 << " side=" << exitSide << "\n";
}

static void cancelEntrySide(InstrumentState& st) {
    if (st.entryActive) {
        requestCancel(st, st.entryId, ReqKind::ENTRY_CANCEL);
    }
}

static void armEntry(InstrumentState& st, long long targetPrice) {
    char entrySide = (st.moverSide == 'B') ? 'S' : 'B';
    long long entryPx = (entrySide == 'S') ? targetPrice + kEntryOffsetTicks * st.meta.ticksize
                                            : targetPrice - kEntryOffsetTicks * st.meta.ticksize;
    entryPx = clampToBand(entryPx, st.meta);
    requestAdd(st, entrySide, entryPx, kEntrySize, ReqKind::ENTRY_ADD);
}

// Proactive entry, independent of the mover's own order flow. Where armEntry
// only fires reactively off onMoverTargetRevealed (i.e. only the instant the
// mover itself trades), this fires directly off the z-score computed every
// bbo tick in updateFv — so a sustained deviation from mean gets traded even
// during a quiet stretch with no mover activity.
//
// Side/price logic is the mirror image of armEntry's: there we knew the
// mover's direction and entered ahead of it. Here we don't have a directional
// signal from the mover — we have a statistical view (fv is |z| stdevs from
// trueMean) — so side is chosen to fade the deviation: price above mean ->
// sell (expect reversion down), price below mean -> buy (expect reversion
// up). Entry price is a marketable limit at the current fv +/- one tick, on
// the side that crosses immediately, matching the same "get filled now"
// intent as armEntry's mover-target-offset pricing.
static void armEntryFromZScore(InstrumentState& st) {
    char entrySide = (st.zScore > 0.0) ? 'S' : 'B';
    long long fvRounded = static_cast<long long>(std::llround(st.fv));
    long long entryPx = (entrySide == 'S') ? fvRounded - kEntryOffsetTicks * st.meta.ticksize
                                            : fvRounded + kEntryOffsetTicks * st.meta.ticksize;
    entryPx = clampToBand(entryPx, st.meta);
    requestAdd(st, entrySide, entryPx, kEntrySize, ReqKind::ENTRY_ADD);
    DiscoveryLog << nowNs() << ",entry_armed_zscore," << st.feed << " z=" << st.zScore << "\n";
}

static void onMoverTargetRevealed(InstrumentState& st, char side, long long price) {
    if (st.entryActive) {
        bool sweptPast = (st.entrySide == 'S' && price > st.entryPrice) ||
                          (st.entrySide == 'B' && price < st.entryPrice);
        if (sweptPast) {
            DiscoveryLog << nowNs() << ",pull," << st.feed << ",target=" << price
                         << " entry=" << st.entryPrice << "\n";
            cancelEntrySide(st);
        }
    }
    st.lastTarget = price;
    st.moverSide = side;
    if (!st.entryActive && !st.haveExit && st.position == 0) {
        armEntry(st, price);
    }
}

static void handleOwnFill(InstrumentState& st, const std::string& sender, const std::string& orderId,
                           long long vol, long long price) {
    if (sender != gSender) return;
    if (orderId == st.entryId && st.entryActive) {
        st.entryFilled += vol;
        st.position += (st.entrySide == 'S') ? -vol : vol;
        st.realizedPnl -= (st.entrySide == 'S' ? -1 : 1) * static_cast<double>(vol) * static_cast<double>(price);
        PnlFile << nowNs() << "," << st.feed << ",entry_fill," << orderId << "," << vol << "," << price
                << "," << st.position << "\n";
        if (st.entryFilled >= kEntrySize) {
            st.entryActive = false;
            postExitLadder(st);
        }
        return;
    }
    for (auto& rung : st.exitLadder) {
        if (rung.id == orderId) {
            rung.filled += vol;
            char exitSide = (st.entrySide == 'S') ? 'B' : 'S';
            st.position += (exitSide == 'S') ? -vol : vol;
            st.realizedPnl += (exitSide == 'S' ? 1 : -1) * static_cast<double>(vol) * static_cast<double>(price);
            PnlFile << nowNs() << "," << st.feed << ",exit_fill," << orderId << "," << vol << "," << price
                    << "," << st.position << "\n";
            break;
        }
    }
    if (st.position == 0 && st.haveExit) {
        bool allDone = true;
        for (auto& r : st.exitLadder) if (r.filled < r.size) allDone = false;
        if (allDone || st.position == 0) {
            for (size_t i = 0; i < st.exitLadder.size(); ++i) {
                auto& r = st.exitLadder[i];
                if (r.filled < r.size) requestCancel(st, r.id, ReqKind::EXIT_CANCEL, static_cast<int>(i));
            }
            st.exitLadder.clear();
            st.haveExit = false;
        }
    }
}

static void handleOrderReply(const std::string& feed, const std::string& reqId, const std::string& payload) {
    auto it = gInstruments.find(feed);
    if (it == gInstruments.end()) return;
    InstrumentState& st = it->second;
    if (!st.awaitingAck || st.inFlight.reqId != reqId) {
        DiscoveryLog << nowNs() << ",stale_reply," << feed << ",reqId=" << reqId << "\n";
        return;
    }
    auto toks = split(payload, ' ');
    bool ok = toks.size() >= 2 && toks[1] == "Y";
    InFlightRequest req = st.inFlight;

    if (req.kind == ReqKind::ENTRY_ADD) {
        if (!ok) {
            DiscoveryLog << nowNs() << ",entry_rejected," << feed << "," << payload << "\n";
            st.entryActive = false;
        } else {
            long long immediateFillVol = toks.size() >= 3 ? std::atoll(toks[2].c_str()) : 0;
            if (immediateFillVol > 0) handleOwnFill(st, gSender, req.orderId, immediateFillVol, req.price);
        }
    } else if (req.kind == ReqKind::EXIT_ADD) {
        if (!ok) {
            DiscoveryLog << nowNs() << ",exit_rung_rejected," << feed << "," << payload << "\n";
            for (auto rit = st.exitLadder.begin(); rit != st.exitLadder.end(); ++rit) {
                if (rit->id == req.orderId) { st.exitLadder.erase(rit); break; }
            }
        } else {
            long long immediateFillVol = toks.size() >= 3 ? std::atoll(toks[2].c_str()) : 0;
            if (immediateFillVol > 0) handleOwnFill(st, gSender, req.orderId, immediateFillVol, req.price);
        }
    } else if (req.kind == ReqKind::ENTRY_CANCEL) {
        DiscoveryLog << nowNs() << ",entry_cancel_ack," << feed << "," << payload << "\n";
        st.entryActive = false;
    } else if (req.kind == ReqKind::EXIT_CANCEL) {
        DiscoveryLog << nowNs() << ",exit_cancel_ack," << feed << "," << payload << "\n";
    }

    st.awaitingAck = false;
    st.inFlight = InFlightRequest{};
    issueNextQueuedAction(st);
}

static void checkTimeout(InstrumentState& st) {
    if (!st.haveExit || st.halfLife <= 0.0) return;
    long long elapsedNs = nowNs() - st.entryTsNs;
    double elapsedSec = static_cast<double>(elapsedNs) / 1e9;
    if (elapsedSec > kTimeoutHalfLives * st.halfLife) {
        DiscoveryLog << nowNs() << ",timeout_escalate," << st.feed << ",position=" << st.position << "\n";
        for (size_t i = 0; i < st.exitLadder.size(); ++i) {
            auto& r = st.exitLadder[i];
            if (r.filled < r.size) requestCancel(st, r.id, ReqKind::EXIT_CANCEL, static_cast<int>(i));
        }
        st.exitLadder.clear();
        st.haveExit = false;
    }
}

static void updateFv(InstrumentState& st) {
    st.fv = fairValue(st.bidPx, st.bidVol, st.askPx, st.askVol, st.meta.ticksize, st.meta.ref_price);
    if (gPhase == Phase::WARMUP) {
        st.warmupMids.push_back(st.fv);
        st.midHist.emplace_back(nowNs(), st.fv);
        if (st.midHist.size() > kMaxHistPoints) st.midHist.pop_front();
    } else if (st.role == Role::DRIVER) {
        st.zScore = computeZScore(st.fv, st.trueMean, st.stdev);
        checkTimeout(st);
        if (!st.entryActive && !st.haveExit && st.position == 0 &&
            std::fabs(st.zScore) >= kZScoreEntryThreshold) {
            armEntryFromZScore(st);
        }
    }
}

static void handleBbo(const std::vector<std::string>& toks) {
    if (toks.size() < 6) return;
    std::string feed = toks[1];
    auto it = gInstruments.find(feed);
    if (it == gInstruments.end()) return;
    InstrumentState& st = it->second;
    st.bidPx = (toks[2] == "-") ? -1 : std::atoll(toks[2].c_str());
    st.bidVol = (toks[3] == "-") ? -1 : std::atoll(toks[3].c_str());
    st.askPx = (toks[4] == "-") ? -1 : std::atoll(toks[4].c_str());
    st.askVol = (toks[5] == "-") ? -1 : std::atoll(toks[5].c_str());
    updateFv(st);
}

static void handleMd(const std::string& feed, const std::vector<std::string>& toks) {
    auto it = gInstruments.find(feed);
    if (it == gInstruments.end()) return;
    InstrumentState& st = it->second;
    if (toks.size() < 2) return;
    const std::string& type = toks[1];

    if (type == "A" && toks.size() >= 6) {
        auto [sender, orderid] = splitId(toks[2]);
        char side = toks[3][0];
        long long price = std::atoll(toks[5].c_str());
        if (gPhase == Phase::LIVE && st.role == Role::DRIVER && sender == st.moverSender) {
            onMoverTargetRevealed(st, side, price);
        }
    } else if (type == "E" && toks.size() >= 8) {
        auto [inSender, inOid] = splitId(toks[2]);
        auto [rsSender, rsOid] = splitId(toks[3]);
        long long vol = std::atoll(toks[4].c_str());
        long long price = std::atoll(toks[5].c_str());
        char aggressorSide = toks[7].empty() ? 0 : toks[7][0];
        if (gPhase == Phase::WARMUP) {
            st.aggressorVolume[inSender] += vol;
            st.totalDriverVolume += vol;
        }
        if (gPhase == Phase::LIVE && st.role == Role::DRIVER && !st.moverSender.empty() &&
            inSender == st.moverSender) {
            onMoverTargetRevealed(st, aggressorSide, price);
        }
        handleOwnFill(st, inSender, inOid, vol, price);
        handleOwnFill(st, rsSender, rsOid, vol, price);
    }
}

static void logRaw(const RawMsg& m, const std::vector<std::string>& toks, const std::string& feed) {
    if (!MDLogFile.is_open()) return;
    std::string wireTs = toks.empty() ? "" : toks[0];
    std::string type = toks.size() > 1 ? toks[1] : "";

    // sender,type,id,vol,price,incoming,resting,matchid,aggressor_side
    std::string sender, id, vol, price, incoming, resting, matchid, aggressorSide;

    if (type == "A" && toks.size() >= 6) {
        // <ts> A <id:17> <B|S> <volume> <price>
        auto [s, oid] = splitId(toks[2]);
        sender = s;
        id = oid;
        vol = toks[4];
        price = toks[5];
    } else if ((type == "E" || type == "T") && toks.size() >= 8) {
        // <ts> E/T <incoming:17> <resting:17> <volume> <price> <matchid> <B|S>
        auto [inSender, inOid] = splitId(toks[2]);
        auto [rsSender, rsOid] = splitId(toks[3]);
        incoming = toks[2];
        resting = toks[3];
        vol = toks[4];
        price = toks[5];
        matchid = toks[6];
        aggressorSide = toks[7];
        // no single "sender"/"id" for a two-party execution; leave blank,
        // incoming/resting carry the full sender:orderid for each side
        (void)inSender; (void)inOid; (void)rsSender; (void)rsOid;
    } else if (type == "C" && toks.size() >= 3) {
        // <ts> C <id:17>
        auto [s, oid] = splitId(toks[2]);
        sender = s;
        id = oid;
    }

    MDLogFile << m.recvNs << "," << wireTs << "," << feed << "," << m.subject << ","
              << sender << "," << type << "," << id << "," << vol << "," << price << ","
              << incoming << "," << resting << "," << matchid << "," << aggressorSide << "\n";
}

static void onRawMsg(natsConnection*, natsSubscription*, natsMsg* msg, void*) {
    RawMsg rm;
    rm.subject = natsMsg_GetSubject(msg);
    rm.payload.assign(natsMsg_GetData(msg), natsMsg_GetDataLength(msg));
    rm.recvNs = nowNs();
    {
        std::lock_guard<std::mutex> lk(gMutex);
        gQueue.push_back(std::move(rm));
    }
    gCv.notify_one();
    natsMsg_Destroy(msg);
}

static void dispatch(const RawMsg& m) {
    auto subjParts = split(m.subject, '.');
    if (!subjParts.empty() && subjParts[0] == "INBOX") {
        if (subjParts.size() >= 4) {
            handleOrderReply(subjParts[2], subjParts[3], m.payload);
        }
        return;
    }
    auto toks = split(m.payload, ' ');
    if (subjParts.size() >= 2 && subjParts[1] == "bbo") {
        handleBbo(toks);
    } else if (subjParts.size() >= 3 && subjParts[1] == "md") {
        std::string feed = subjParts[2];
        logRaw(m, toks, feed);
        handleMd(feed, toks);
    }
}

// Driver identification, per mem2.md's settled design: the driver is the
// instrument with ONE dominant aggressor (the mover) sweeping its own book
// via marketable orders — NOT the instrument that "leads" cross-instrument
// price correlation. Piggyback requotes are triggered by an integer-rounding
// threshold on fv(driver), which is a nonlinear relationship that fixed-lag
// Pearson correlation (the old bestFit/UnionFind clustering) cannot reliably
// detect, and which has no reason to make the driver look "earlier" in a
// lagged-correlation sense than its own piggybacks (AAH6 updates far more
// often than AAM6/AAU6 simply because the mover ticks every 0.2s while
// piggybacks only requote on integer-crossing — that update-rate asymmetry
// can swamp any genuine lead/lag signal). Applying the dominance check
// uniformly (not just to singleton clusters) avoids picking a driver based
// on noisy lag correlation and then never sanity-checking it.
static void runDiscovery() {
    std::vector<std::string> feeds;
    for (auto& kv : gInstruments) feeds.push_back(kv.first);

    std::cout << "[DISCOVERY] warmup complete, analyzing " << feeds.size() << " instruments" << std::endl;

    // Step 1: for every instrument, find its dominant aggressor and the
    // share of total traded volume that aggressor accounts for. Only an
    // instrument with a single sender responsible for most of its volume
    // is a driver candidate.
    struct DominanceInfo {
        std::string moverSender;
        long long bestVol = -1;
        long long totalVol = 0;
        double share = 0.0;
        bool dominant = false;
    };
    std::unordered_map<std::string, DominanceInfo> dom;

    for (auto& feed : feeds) {
        InstrumentState& st = gInstruments[feed];
        DominanceInfo info;
        info.totalVol = st.totalDriverVolume;
        for (auto& kv : st.aggressorVolume) {
            if (kv.second > info.bestVol) { info.bestVol = kv.second; info.moverSender = kv.first; }
        }
        info.share = info.totalVol > 0 && info.bestVol > 0
                         ? static_cast<double>(info.bestVol) / static_cast<double>(info.totalVol)
                         : 0.0;
        info.dominant = info.totalVol > 0 && info.share >= kMoverDominanceShare;
        dom[feed] = info;
        DiscoveryLog << nowNs() << ",dominance_check," << feed << ",mover=" << info.moverSender
                     << " share=" << info.share << " dominant=" << (info.dominant ? 1 : 0) << "\n";
        std::cout << "[DISCOVERY] " << feed << ": top aggressor=" << info.moverSender
                  << " share=" << info.share << " (" << st.aggressorVolume.size()
                  << " distinct aggressors, total_vol=" << info.totalVol << ") "
                  << (info.dominant ? "-> DRIVER candidate" : "-> not dominant") << std::endl;
    }

    // Step 2: candidate drivers are the dominant instruments. Everything
    // else is provisionally a piggyback of *some* driver — determined by
    // shared reference to the same underlying (feed prefix), which is a
    // structural fact given at instrument-discovery time (EX_META), not
    // inferred from noisy price correlation. This matches the sim's own
    // mechanics: piggybacks key their center off fv(one specific driver),
    // and same-underlying feeds share the driver.
    std::vector<std::string> drivers;
    for (auto& feed : feeds) {
        if (dom[feed].dominant) drivers.push_back(feed);
    }

    if (drivers.empty()) {
        std::cout << "[DISCOVERY][WARN] no dominant-aggressor instrument found in "
                  << feeds.size() << " instruments; leaving all UNKNOWN" << std::endl;
        for (auto& feed : feeds) {
            gInstruments[feed].role = Role::UNKNOWN;
            DiscoveryLog << nowNs() << ",role_assigned," << feed << ",UNKNOWN(no_driver_found)\n";
        }
        return;
    }

    int clusterId = 0;
    for (auto& driverFeed : drivers) {
        std::string prefix = driverFeed.substr(0, 2);
        InstrumentState& dst = gInstruments[driverFeed];
        dst.role = Role::DRIVER;
        dst.cluster = clusterId;
        dst.moverSender = dom[driverFeed].moverSender;
        dst.trueMean = meanOf(dst.warmupMids);
        dst.stdev = computeStdev(dst.warmupMids, dst.trueMean);
        dst.halfLife = computeHalfLife(dst.warmupMids);

        DiscoveryLog << nowNs() << ",role_assigned," << driverFeed << ",DRIVER\n";
        DiscoveryLog << nowNs() << ",mover_identified," << driverFeed << "," << dst.moverSender
                     << " mean=" << dst.trueMean << " stdev=" << dst.stdev << " halfLife=" << dst.halfLife << "\n";
        std::cout << "[DISCOVERY] cluster " << clusterId << " driver: " << driverFeed
                  << " mover=" << dst.moverSender << " mean=" << dst.trueMean
                  << " stdev=" << dst.stdev << " halfLife=" << dst.halfLife << "s" << std::endl;

        for (auto& feed : feeds) {
            if (feed == driverFeed) continue;
            if (feed.substr(0, 2) != prefix) continue;
            if (dom[feed].dominant) continue;  // it's its own driver candidate, not a piggyback
            InstrumentState& pst = gInstruments[feed];
            pst.role = Role::PIGGYBACK;
            pst.cluster = clusterId;
            DiscoveryLog << nowNs() << ",role_assigned," << feed << ",PIGGYBACK(of " << driverFeed << ")\n";
            std::cout << "[DISCOVERY] cluster " << clusterId << ": " << feed
                      << " -> PIGGYBACK (of " << driverFeed << ")" << std::endl;
        }
        clusterId++;
    }

    // Anything sharing a prefix with a driver but not yet assigned, and
    // anything with no driver at all for its prefix, is left UNKNOWN rather
    // than guessed at.
    for (auto& feed : feeds) {
        InstrumentState& st = gInstruments[feed];
        if (st.role == Role::UNKNOWN) {
            DiscoveryLog << nowNs() << ",role_assigned," << feed << ",UNKNOWN(no_driver_for_prefix)\n";
            std::cout << "[DISCOVERY] " << feed << " -> UNKNOWN (no driver identified for prefix "
                      << feed.substr(0, 2) << ")" << std::endl;
        }
    }
}

static void mainLoop() {
    auto warmupEnd = std::chrono::steady_clock::now() + kWarmupPeriod;
    std::cout << "warming up (clustering + half-life + mover ID)..." << std::endl;

    while (gPhase == Phase::WARMUP && !gShuttingDown.load()) {
        std::unique_lock<std::mutex> lk(gMutex);
        bool got = gCv.wait_until(lk, std::chrono::steady_clock::time_point(warmupEnd),
                                   [] { return !gQueue.empty() || gShuttingDown.load(); });
        std::vector<RawMsg> batch;
        while (!gQueue.empty()) { batch.push_back(std::move(gQueue.front())); gQueue.pop_front(); }
        lk.unlock();
        for (auto& m : batch) dispatch(m);
        (void)got;
        flushAllLogs();
        if (gShuttingDown.load()) break;
        if (std::chrono::steady_clock::now() >= warmupEnd) {
            runDiscovery();
            gPhase = Phase::LIVE;
            std::cout << "live." << std::endl;
        }
    }

    while (!gShuttingDown.load()) {
        std::unique_lock<std::mutex> lk(gMutex);
        gCv.wait_for(lk, std::chrono::milliseconds(200),
                     [] { return !gQueue.empty() || gShuttingDown.load(); });
        std::vector<RawMsg> batch;
        while (!gQueue.empty()) { batch.push_back(std::move(gQueue.front())); gQueue.pop_front(); }
        lk.unlock();
        for (auto& m : batch) dispatch(m);
        for (auto& [feed, st] : gInstruments) {
            if (st.role == Role::DRIVER) checkTimeout(st);
            // Watchdog: async means there's no built-in timeout on a reply
            // like the old blocking sendRequest had. If a request's ack
            // never arrives (dropped reply, exchange hiccup), awaitingAck
            // would otherwise stay true forever and this instrument goes
            // permanently silent — no new entries, no queued follow-ups
            // ever fire again. Treat a stale in-flight request as failed:
            // clear the busy flag, log it, and let the next queued action
            // (or next natural trigger) proceed.
            if (st.awaitingAck && (nowNs() - st.inFlight.sentNs) > kOrderTimeoutMs * 1'000'000LL) {
                DiscoveryLog << nowNs() << ",request_timeout," << feed << ",kind="
                             << static_cast<int>(st.inFlight.kind) << " id=" << st.inFlight.orderId << "\n";
                std::cout << "[WATCHDOG] " << feed << " request " << st.inFlight.orderId
                          << " never acked after " << kOrderTimeoutMs << "ms, giving up on it" << std::endl;
                st.awaitingAck = false;
                st.inFlight = InFlightRequest{};
                issueNextQueuedAction(st);
            }
        }
        // Flush every tick (~200ms) rather than relying on process exit,
        // so a hard kill / crash loses at most one tick of logs, not the
        // whole buffered run.
        flushAllLogs();
    }

    std::cout << "shutdown signal received, flushing logs and exiting" << std::endl;
    flushAllLogs();
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::signal(SIGTERM, onSignal);
    std::signal(SIGINT, onSignal);

    PnlFile.open(envOr("PNL_LOGGING", "/data/pnl.csv"));
    if (!PnlFile.is_open()) {
        std::cerr << "failed to open pnl file\n";
        return 1;
    }
    PnlFile << "ts,feed,event,order_id,vol,price,position\n";

    DiscoveryLog.open(envOr("DISCOVERY_LOGGING", "/data/discovery.csv"));
    if (DiscoveryLog.is_open()) DiscoveryLog << "ts,event,feed_or_pair,detail\n";

    openMDLog();

    const char* natsUrl = envOr("NATS_URL", "nats://localhost:4222");
    gSender = envOr("SENDER", "QUOTE01");

    std::cout << "quoter connected to " << natsUrl << " as " << gSender << std::endl;

    natsStatus s = NATS_OK;
    jsCtx* js = nullptr;
    kvStore* metakv = nullptr;

    s = natsConnection_ConnectTo(&gConn, natsUrl);
    if (s != NATS_OK) {
        std::cout << "connection failed: " << natsStatus_GetText(s) << std::endl;
        return 1;
    }
    std::cout << "Connected!" << std::endl;

    s = natsConnection_JetStream(&js, gConn, nullptr);
    if (s == NATS_OK) s = js_KeyValue(&metakv, js, "EX_META");
    if (s != NATS_OK) {
        std::cout << "failed to connect to EX_META: " << natsStatus_GetText(s) << std::endl;
        return 1;
    }

    kvKeysList kvkeyslist;
    memset(&kvkeyslist, 0, sizeof(kvkeyslist));
    s = kvStore_Keys(&kvkeyslist, metakv, nullptr);
    if (s != NATS_OK) {
        std::cout << "failed to list EX_META keys: " << natsStatus_GetText(s) << std::endl;
        return 1;
    }
    std::cout << "discovered " << kvkeyslist.Count << " instruments" << std::endl;

    for (int i = 0; i < kvkeyslist.Count; i++) {
        std::string feed = kvkeyslist.Keys[i];
        kvEntry* entry = nullptr;
        s = kvStore_Get(&entry, metakv, feed.c_str());
        if (s != NATS_OK) {
            std::cout << "couldn't get kv for " << feed << ": " << natsStatus_GetText(s) << std::endl;
            s = NATS_OK;
            continue;
        }
        std::string metaValue = kvEntry_ValueString(entry);
        std::cout << "[EX_META " << feed << "] " << metaValue << std::endl;
        kvEntry_Destroy(entry);

        InstrumentState st;
        st.feed = feed;
        st.meta = parseMeta(metaValue);
        gInstruments[feed] = st;
    }
    kvKeysList_Destroy(&kvkeyslist);

    natsSubscription* bboSub = nullptr;
    natsStatus subStatus = natsConnection_Subscribe(&bboSub, gConn, "ex.bbo.>", onRawMsg, nullptr);
    std::cout << (subStatus == NATS_OK ? "subbed to BBO" : "failed to sub BBO") << std::endl;

    natsSubscription* mdSub = nullptr;
    subStatus = natsConnection_Subscribe(&mdSub, gConn, "ex.md.>", onRawMsg, nullptr);
    std::cout << (subStatus == NATS_OK ? "subbed to MD" : "failed to sub MD") << std::endl;

    natsSubscription* replySub = nullptr;
    std::string replyWildcard = "INBOX." + gSender + ".>";
    subStatus = natsConnection_Subscribe(&replySub, gConn, replyWildcard.c_str(), onRawMsg, nullptr);
    std::cout << (subStatus == NATS_OK ? "subbed to order replies" : "failed to sub order replies") << std::endl;

    mainLoop();

    return 0;
}