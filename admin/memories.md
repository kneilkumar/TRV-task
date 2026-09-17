#include <nats.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// logs
//==============================
static std::ofstream PnlFile;
static std::ofstream DiscoveryLog;
static std::ofstream MDLogFile;
const std::chrono::seconds half_life_collection_period{60};

//==============================



// global constants + vars + structs go here
//==============================

static std::string gSender;
static natsConnection* gConn = nullptr;
static std::map<std::string, InstrumentState> gInstruments;


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


struct InstrumentState {
    long long bix_px = -1, ask_px = -1, bid_vol = -1, ask_vol = 1;

    std::string isDriver;
    std::string moverSender;
    std::string feed;
    InstrumentMeta meta;

    double mp_samples[20];
    int mp_sample_index = 0;

    double true_fv = 0.0;
    int rolling_window_size = 0;
    double std = 0.0;
    double fv_est = 0.0;
    double half_life = 0.0;
    double z_score = 0.0;
    double samples_added = 0.0;
};


//==============================


// helpers go here (maths for mean reversion etc)
//==============================

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

static const char* envOr(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && v[0 != '\0']) ? v : fallback;
}

static void openMDLog() {
    MDLogFile.open(envOr("MD_LOGGING", "/data/md.csv"));
    if (MDLogFile.is_open()) MDLogFile << "recv_ns,wire_ts,feed,subject,sender,type,id,vol,price,incoming,resting,matchid,aggressor_side\n";
}

static InstrumentMeta parseMeta(const std::string& value) {
    InstrumentMeta m;
    for (auto&kv : split(value, ' ')) {
        auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        std::string k = kv.substr(0,eq);
        long long v = std::atoll(kv.substr(eq+1).c_str());
        if (k == "ticksize") m.ticksize = v;
        else if (k == "ref_price") m.ref_price = v;
        else if (k == "min_vol") m.min_vol = v;
        else if (k == "max_vol") m.max_vol = v;
        else if (k == "max_tps") m.max_tps = v;
        else if (k == "pos_lim") m.pos_lim = v;
        else if (k == "band") m.band = v;
        
    }
    return m;
}

std::vector<double> shift_elements(const std::vector<double>& vec) {
    int vec_len = vec.size();
    std::vector<double> shifted_vector(vec_len);
    for (int i = 1; i < vec_len; i++) {
        shifted_vector[i] = vec[i-1];
    }
    return shifted_vector;
}

void addsample(InstrumentState& state, double sample, int window_size) {
    state.mp_samples[state.mp_sample_index] = sample;
    state.mp_sample_index = (state.mp_sample_index + 1) % window_size;
    if (state.samples_added < 20) state.samples_added++;
}


//==============================

// nats messaging infra
//==============================

struct RawMsg {
    long long seq = 0;
    std::string subject;
    std::string payload;
    std::chrono::steady_clock::time_point recvtime;
};


// mean rev stuff
//==============================
int recover_half_life(const std::vector<double>& midprice_series) {
    int shift_size = 1;
    std::vector<double> shifted_midprice;
    std::vector<double> mp_series_copy = midprice_series;
    shifted_midprice = shift_elements(midprice_series);
    shifted_midprice.erase(shifted_midprice.begin());
    mp_series_copy.erase(mp_series_copy.begin());
    
    double dxdy = 0;
    double dxdx = 0;

    double xbar = std::accumulate(shifted_midprice.begin(), shifted_midprice.end(), 0);
    double ybar = std::accumulate(mp_series_copy.begin(), mp_series_copy.end(), 0);

    for (int i=0;i<shifted_midprice.size();i++) {
        dxdy = dxdy + (shifted_midprice[i] - xbar)*(mp_series_copy[i] - ybar);
        dxdx = dxdx + (shifted_midprice[i] - xbar)*(shifted_midprice[i] - xbar);
    }

    double fi = dxdy/dxdx;

    double half_life = -1*std::log(2)/std::log(fi);

    return half_life;

}

int recover_zscore(int& window_size, InstrumentState& state) {
    double inner_sum = 0.0;
    double sample_mean = 0.0;
    for (int i=0; i < window_size;i++) {
        sample_mean = inner_sum + state.mp_samples[i];
    }
    sample_mean = sample_mean/window_size;

    for (int i=0; i < window_size;i++) {
        inner_sum = inner_sum + ((state.mp_samples[i] - sample_mean)*(state.mp_samples[i] - sample_mean));
    }
    inner_sum = inner_sum/(window_size - 1);

    double std = std::sqrt(inner_sum);
    state.std = std;

    if (std < 1e-9) return 0.0;

    double z_score = (state.fv_est - state.true_fv) / state.std;
}

//==============================





//==============================


// trading loop
//==============================
int mainLoop() {

    // sit idle absorbing data for half life etc (assume it is stationary)
    auto warmup_end = std::chrono::steady_clock::now() + half_life_collection_period;
    std::cout << "collecting mid prices for half life calculation" << std::endl;

    while (std::chrono::steady_clock::now() <= warmup_end) {
        

    }

    // get all the mean reversion parameters per instrument in a cluster

    // compute the quotes

    // submit them

    // cancellation logic

}


//==============================



//main 
//==============================

int main(int argc, char**argv){
    (void)argc;
    (void)argv;

    setvbuf(stdout, nullptr, _IOLBF, 0);

    PnlFile.open(envOr("PNL_LOGGING", "/data/pnl.csv"));
    
    if (!PnlFile.is_open()) {
        std::cerr << "failed to open pnl file\n";
        return 1;
    }

    DiscoveryLog.open(envOr("DISCOVERY_LOGGING", "/data/discovery.csv"));
    if (DiscoveryLog.is_open()) DiscoveryLog << "ts,event,feed_or_pair,detail\n";

    openMDLog();

    const char* natsUrl = envOr("NATS_URL", "nats://local_host:4222");
    gSender = envOr("SENDER", "QUOTE01");

    std::cout << "quoter connected to " << natsUrl << " as " << gSender << std::endl;

    natsStatus s = NATS_OK;
    jsCtx* js = nullptr;
    kvStore* metakv = nullptr;

    s = natsConnection_ConnectTo(&gConn, natsUrl);
    if (s != NATS_OK){
        std::cout << "connection failed: " << natsStatus_GetText(s) << "\n" <<std::endl;
        return 1;
    }
    std::cout << "Connected!\n" << std::endl;

    s = natsConnection_JetStream(&js, gConn, nullptr);
    if (s == NATS_OK) s = js_KeyValue(&metakv, js ,"EX_META");
    if (s != NATS_OK) {
        std::cout << "failed to connected to EX_META!\n" << natsStatus_GetText(s) << std::endl; 
        return 1;
    }

    kvKeysList kvkeyslist;
    memset(&kvkeyslist, 0, sizeof(kvkeyslist));
    s = kvStore_Keys(&kvkeyslist, metakv, nullptr);
    if (s != NATS_OK) {
        std::cout << "failed to store keys!\n" << natsStatus_GetText(s) << std::endl; 
        return 1;
    }
    std::cout << "discovered " << kvkeyslist.Count << " in bucket!" << std::endl; 

    for (int i = 0; i < kvkeyslist.Count; i++ ){
        std::string feed = kvkeyslist.Keys[i];
        kvEntry* entry = nullptr;
        s = kvStore_Get(&entry, metakv, feed.c_str());
        if (s != NATS_OK) {
            std::cout << "couldn't get kv for " << feed.c_str() << "\nSee error\n" <<  natsStatus_GetText(s) << std::endl; 
            s = NATS_OK;
            continue;
        }
        std::string metaValue = kvEntry_ValueString(entry);
        std::cout << "[EX_META " << feed.c_str() << "] " << metaValue.c_str() << std::endl;
        kvEntry_Destroy(entry);

        InstrumentState st;
        st.feed = feed;
        st.meta = parseMeta(metaValue);
        gInstruments[feed] = st;
    }
    kvKeysList_Destroy(&kvkeyslist);

    natsSubscription* bboSub = nullptr;
    natsStatus subStatus = natsConnection_Subscribe(&bboSub, gConn, "ex.bbo.>", onRawMsg, nullptr);
    if (subStatus == NATS_OK) std::cout << "subbed to Nats BBO! " << std::endl;
    else std::cout << "failed to connect to nats bbo! " <<std::endl;

    natsSubscription* mdSub = nullptr;
    subStatus = natsConnection_Subscribe(&mdSub, gConn, "ex.md.>", onRawMsg, nullptr);
    (subStatus == NATS_OK) ? std::cout << "subbed to Nats MD! " << std::endl : std::cout << "failed to connect to nats bbo! " <<std::endl;

    mainLoop();

    // Shutdown Handling goes here

}

