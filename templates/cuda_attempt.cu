#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <numeric>
#include <functional>

#define n_nodes {{n_nodes}}
#define num_states {{num_states}}

extern "C" {
__host__ __device__ void changebase(int number, std::vector<unsigned int>& state){

    int current_state = num_states - 1;
    int Number = number;
    int NumStates = num_states;
    int quotient = (int)Number/NumStates;
    int remainder = Number % NumStates;

    Number = quotient;
    state[current_state] = remainder;

    current_state = current_state - 1;

    while (quotient !=0){
        quotient = (int)Number/NumStates;
        remainder = Number % NumStates;
        state[current_state] = remainder;
        Number = quotient;
        current_state = current_state - 1;
        };
}


__host__ __device__ void update(std::vector<unsigned int> x, std::vector<unsigned int>& vect){
{{functions}}

    std::transform(vect.begin(),vect.end(),vect.begin(),std::bind2nd(std::modulus<unsigned int>(),3));
    };


void print_vect(int temp_vect){
    for (const unsigned int& i : temp_vect){
        std::cout << i << ' ';
        };
    std::cout << '\n';
}


__host__ __device__ void to_string(std::vector<unsigned int> v, std::string& ss ){
    ss = std::accumulate(std::next(v.begin()), v.end(),
        std::to_string(v[0]), // start with first element
        [](std::string a, unsigned int b) {return a + std::to_string(b);});
}

__global__ void AttractorFinder(std::string* result){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

//    std::vector<unsigned int> start_vect(n_nodes, 0);
//    std::vector<unsigned int> end_vect(n_nodes, 0);

    int start_vect[n_nodes];
    int end_vect[n_nodes];

    std::string end_string;
    std::set<std::vector<unsigned int>> EmptySet;

    std::set<std::vector<unsigned int>> visited_states;


    changebase(tid, start_vect);

    while(true){

        update(start_vect, end_vect);
        if (visited_states.count(end_vect)==1){break;}
        visited_states.insert(end_vect);
        start_vect = end_vect;

    };
    to_string(end_vect, end_string);
    result[tid] = end_string;
};
}