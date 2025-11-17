#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <array>
#include <cmath>
#include <functional>
#include <utility>
#include <sstream>

// Calculate the player's optimal expected gain (in [-1,1]) from a given hand score
// against a dealer upcard. The function returns the max between standing and
// hitting (the hit EV is the average over possible next cards). Dealer is
// modeled with an infinite deck (cards 1..10 uniform) and stands on all 17s.
// Payoffs: win=+1, push=0, loss=-1.
// Options returned for a given player sum and dealer upcard
struct Options {
    double standWin;
    double standLoss;
    double hitWin;
    double hitLoss;
    double optWin;
    double optLoss;
    std::string bestAction;
};

// Compute detailed stand/hit/optimal win/loss probabilities.
// Uses on-the-fly memoized dealer comparison (no full distribution precompute).
Options computeOptions(int hand_score, int dealer_upcard) {
    const int TARGET = 21;
    const int MAX_CARD = 10;
    const int MAX_SUM = TARGET + MAX_CARD;

    Options out{};
    if (dealer_upcard < 1 || dealer_upcard > 10) return out;

    // Dealer memo tables: bust, less, equal (per start sum and player total)
    std::vector<std::vector<double>> dealerBust(MAX_SUM + 1, std::vector<double>(TARGET + 1, -1.0));
    std::vector<std::vector<double>> dealerLess(MAX_SUM + 1, std::vector<double>(TARGET + 1, -1.0));
    std::vector<std::vector<double>> dealerEqual(MAX_SUM + 1, std::vector<double>(TARGET + 1, -1.0));

    std::function<void(int,int)> computeDealer = [&](int s, int pt) {
        if (s > MAX_SUM) return;
        if (dealerBust[s][pt] >= -0.5) return;

        if (s > TARGET) {
            dealerBust[s][pt] = 1.0;
            dealerLess[s][pt] = 0.0;
            dealerEqual[s][pt] = 0.0;
            return;
        }
        if (s >= 17) {
            dealerBust[s][pt] = 0.0;
            dealerLess[s][pt] = (s < pt) ? 1.0 : 0.0;
            dealerEqual[s][pt] = (s == pt) ? 1.0 : 0.0;
            return;
        }

        double accBust = 0.0, accLess = 0.0, accEqual = 0.0;
        for (int v = 1; v <= MAX_CARD; ++v) {
            int ns = s + v;
            if (ns > MAX_SUM) {
                accBust += 1.0;
            } else {
                if (dealerBust[ns][pt] < -0.5) computeDealer(ns, pt);
                accBust += dealerBust[ns][pt];
                accLess += dealerLess[ns][pt];
                accEqual += dealerEqual[ns][pt];
            }
        }
        dealerBust[s][pt] = accBust / double(MAX_CARD);
        dealerLess[s][pt] = accLess / double(MAX_CARD);
        dealerEqual[s][pt] = accEqual / double(MAX_CARD);
    };

    auto standProbs = [&](int s) -> std::pair<double,double> {
        if (s > TARGET) return {0.0, 1.0};
        double win = 0.0, loss = 0.0;
        for (int hole = 1; hole <= MAX_CARD; ++hole) {
            int start = dealer_upcard + hole;
            if (start > MAX_SUM) {
                win += 1.0; // dealer bust
            } else {
                if (dealerBust[start][s] < -0.5) computeDealer(start, s);
                double b = dealerBust[start][s];
                double l = 1.0 - (dealerBust[start][s] + dealerLess[start][s] + dealerEqual[start][s]);
                // win when dealer busts or dealer < s
                win += b + dealerLess[start][s];
                loss += l;
            }
        }
        win /= double(MAX_CARD);
        loss /= double(MAX_CARD);
        return {win, loss};
    };

    // memo for optimal play
    std::vector<double> memoWin(MAX_SUM + 1, -1.0);
    std::vector<double> memoLoss(MAX_SUM + 1, -1.0);

    std::function<std::pair<double,double>(int)> optimal = [&](int s) -> std::pair<double,double> {
        if (s > TARGET) return {0.0, 1.0};
        if (s <= MAX_SUM && memoWin[s] >= -0.5) return {memoWin[s], memoLoss[s]};

        auto stand = standProbs(s);

        double hitW = 0.0, hitL = 0.0;
        for (int v = 1; v <= MAX_CARD; ++v) {
            int ns = s + v;
            auto res = optimal(ns);
            hitW += res.first;
            hitL += res.second;
        }
        hitW /= double(MAX_CARD);
        hitL /= double(MAX_CARD);

        // choose action that maximizes win probability
        if (stand.first >= hitW) {
            if (s <= MAX_SUM) { memoWin[s] = stand.first; memoLoss[s] = stand.second; }
            return {stand.first, stand.second};
        } else {
            if (s <= MAX_SUM) { memoWin[s] = hitW; memoLoss[s] = hitL; }
            return {hitW, hitL};
        }
    };

    // compute for requested hand_score
    auto opt = optimal(hand_score);
    out.optWin = opt.first;
    out.optLoss = opt.second;
    out.bestAction = (out.optWin == -1.0) ? "bust" : "optimal";

    // stand immediate
    auto st = standProbs(hand_score);
    out.standWin = st.first;
    out.standLoss = st.second;

    // hit immediate (average over next-card results, following optimal afterwards)
    double hitW = 0.0, hitL = 0.0;
    for (int v = 1; v <= MAX_CARD; ++v) {
        auto r = optimal(hand_score + v);
        hitW += r.first; hitL += r.second;
    }
    out.hitWin = hitW / double(MAX_CARD);
    out.hitLoss = hitL / double(MAX_CARD);

    // determine best action label comparing stand vs hit immediate (not comparing optimal policy)
    if (out.standWin > out.hitWin) out.bestAction = "stand";
    else if (out.hitWin > out.standWin) out.bestAction = "hit";
    else out.bestAction = "equal";

    return out;
}

// Wrapper kept for name compatibility: returns pair(win,loss) under optimal policy
std::pair<double,double> calculateHitChance(int hand_score, int dealer_upcard) {
    Options o = computeOptions(hand_score, dealer_upcard);
    return {o.optWin, o.optLoss};
}

int main() {
    // Write CSV with options for player scores 4..21 and dealer upcards 1..10
    std::ofstream out("results.csv");
    out << "player_score,dealer_upcard,stand_win,stand_loss,stand_win_loss_ratio,hit_win,hit_loss,hit_win_loss_ratio,best_action,opt_win,opt_loss,opt_win_loss_ratio\n";
    for (int ps = 4; ps <= 21; ++ps) {
        for (int du = 1; du <= 10; ++du) {
            Options o = computeOptions(ps, du);
            auto ratio = [&](double w, double l) {
                if (l <= 0.0) return std::string("inf");
                std::ostringstream ss; ss << std::fixed << std::setprecision(6) << (w / l);
                return ss.str();
            };
            out << ps << "," << du << ","
                << std::fixed << std::setprecision(6) << o.standWin << ","
                << o.standLoss << ","
                << ratio(o.standWin, o.standLoss) << ","
                << o.hitWin << ","
                << o.hitLoss << ","
                << ratio(o.hitWin, o.hitLoss) << ","
                << o.bestAction << ","
                << o.optWin << ","
                << o.optLoss << ","
                << ratio(o.optWin, o.optLoss) << "\n";
        }
    }
    out.close();
    std::cout << "Wrote results.csv (player scores 4..21 vs dealer upcards 1..10)\n";
    return 0;
}