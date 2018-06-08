#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <thread>

typedef std::chrono::high_resolution_clock Clock;
typedef std::vector<unsigned int> Vect;
typedef std::set<std::vector<unsigned int>> VectSet;

int n_nodes = {{n_nodes}};
int num_states = {{num_states}};

VectSet v_states;

void print_vect(Vect temp_vect){
    for (auto& i : temp_vect){
        std::cout << i << ' ';
        };
    std::cout << '\n';
}

void changebase(int number, Vect& state){

    int current_state = num_states -1;
    int Number = number;

    int quotient = (int)Number/num_states;
    int remainder = Number % num_states;

    Number = quotient;
    state[current_state] = remainder;
    current_state = current_state - 1;

    while (quotient !=0){
        quotient = (int)Number/num_states;
        remainder = Number % num_states;
        state[current_state] = remainder;
        Number = quotient;
        current_state = current_state - 1;
        };
}


void update(const Vect& x, Vect& vect){
{{functions}}
    std::transform(vect.begin(),vect.end(),vect.begin(),std::bind2nd(std::modulus<unsigned int>(),3));
    };


void to_string(const Vect& v, std::string& ss ){
    ss = std::accumulate(std::next(v.begin()), v.end(),
        std::to_string(v[0]), // start with first element
        [](std::string a, int b) {return a + std::to_string(b);}
    );
}

void print_set(const VectSet& SetExample){
    std::string state_string;
    for (auto& i : SetExample){
        to_string(i, state_string);
        std::cout << state_string << std::endl;
    }
}

void find_attractor(const int local_count, Vect& e_vect){

    Vect s_vect(n_nodes, 0);

    changebase(local_count, s_vect);
    v_states.clear();

    while(true){
            update(s_vect, e_vect);
            if (v_states.count(e_vect)==1){break;}
            v_states.insert(e_vect);
            s_vect = e_vect;
        };

}

void check(int c, VectSet& FinalSet, Vect& final_vect ){
    find_attractor(c, final_vect);
    FinalSet.insert(final_vect);
}

void run_new(){
    auto total_states = pow(num_states, n_nodes);
    total_states = 1000000;
    VectSet FinalSet;
    Vect final_vect(n_nodes, 0);

    auto t1 = Clock::now();

    for (int count = 0; count < total_states; count++ ){
        check(count, FinalSet, final_vect);
        }

    auto t2 = Clock::now();

    std::cout << "Number of states = " << FinalSet.size() << "\n";
    std::chrono::duration<double, std::milli> total_time = t2 - t1;
    std::cout << "Delta t2-t1: " << total_time.count() << " milliseconds" << std::endl;

    auto t_states = pow(num_states, n_nodes);
    auto rate = 1./(total_states/total_time.count()); // gets us to millisec/state
    rate = rate/1000./60./60.;  // gets us to hour/state
    auto exp_time = rate*t_states; // min/state * state = mins

    std::cout << "Total expected time = " << exp_time << "\n";
    std::cout << "final set \n";
    print_set(FinalSet);
    std::cout << "\n";

}



void run_2(){

    auto total_states = pow(num_states, n_nodes);
    total_states = 100000;
    Vect start_vect(n_nodes, 0);
    auto end_vect = start_vect;
    VectSet FinalSet;
    auto visited_states = FinalSet;

    auto t1 = Clock::now();
    for (auto count = 0; count < total_states; count++ ){
        changebase(count, start_vect);
        if (FinalSet.count(start_vect)==1){continue;}
        visited_states.clear();
        while(true){
            update(start_vect, end_vect);
            if (visited_states.count(end_vect)==1){break;}
            visited_states.insert(end_vect);
            start_vect = end_vect;
        };
        FinalSet.insert(end_vect);
        }

    auto t2 = Clock::now();

    std::cout << "Number of states = " << FinalSet.size() << "\n";
    std::chrono::duration<double, std::milli> total_time = t2 - t1;

    std::cout << "Delta t2-t1: " << total_time.count() << " milliseconds" << std::endl;

    auto t_states = pow(num_states, n_nodes);
    auto rate = 1./(total_states/total_time.count()); // gets us to millisec/state
    rate = rate/1000./60./60.;  // gets us to hour/state
    auto exp_time = rate*t_states; // min/state * state = mins


    std::cout << "Total expected time = " << exp_time << "\n";

    std::cout << "final set \n";
    print_set(FinalSet);
    std::cout << "\n";



};


int main(){
//    run_new();
    run_2();

};
