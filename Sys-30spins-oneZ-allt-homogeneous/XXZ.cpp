#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <complex>
#include <algorithm>
#include "../Eigen/Eigenvalues"
using namespace std;

int N = 30;
double gamma_param = 0.0;
const complex<double> im(0, 1);
Eigen::Matrix2cd Lowering = (Eigen::Matrix2cd() << 0, 0, 1, 0).finished();
Eigen::Matrix2cd Raising = (Eigen::Matrix2cd() << 0, 1, 0, 0).finished();
Eigen::Matrix2cd PauliI = (Eigen::Matrix2cd() << 1, 0, 0, 1).finished();
Eigen::Matrix2cd PauliX = (Eigen::Matrix2cd() << 0, 1, 1, 0).finished();
Eigen::Matrix2cd PauliY = (Eigen::Matrix2cd() << 0, -im, im, 0).finished();
Eigen::Matrix2cd PauliZ = (Eigen::Matrix2cd() << 1, 0, 0, -1).finished();

#define WARM_UP_ENGINE

using vec = tuple<double,double,double> ;

const auto x = [] ( vec v ) { return get<0>(v) ; } ;
const auto y = [] ( vec v ) { return get<1>(v) ; } ;
const auto z = [] ( vec v ) { return get<2>(v) ; } ;

double norm( vec v ) { return sqrt( x(v)*x(v) + y(v)*y(v) + z(v)*z(v) ) ; }

vec random_vector()
{
    static mt19937 twister( 13131 ) ;
    static uniform_real_distribution<double> distr( -1000, 1000 ) ;

    return vec( distr(twister), distr(twister), distr(twister) ) ;
}

vec random_unit_vector()
{
    auto v = random_vector() ;
    constexpr double epsilon = 0.01 ;
    double m = norm(v) ;
    if( m > epsilon ) return vec( x(v)/m, y(v)/m, z(v)/m ) ;
    else return random_unit_vector() ;
}

vector<tuple<double, Eigen::VectorXd> > compute_eig(Eigen::MatrixXd X){
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver;
    eigensolver.compute(X);

    Eigen::VectorXd eigen_values = eigensolver.eigenvalues().real();
    Eigen::MatrixXd eigen_vectors = eigensolver.eigenvectors().real();
    vector<tuple<double, Eigen::VectorXd>> eigen_vectors_and_values;
    for(int i = 0; i < eigen_values.size(); i++){
        tuple<double, Eigen::VectorXd> vec_and_val(eigen_values[i], eigen_vectors.col(i));
        eigen_vectors_and_values.push_back(vec_and_val);
    }
    sort(eigen_vectors_and_values.begin(), eigen_vectors_and_values.end(),
        [&](const tuple<double, Eigen::VectorXd>& a, const tuple<double, Eigen::VectorXd>& b) -> bool{
            return (get<0>(a) < get<0>(b));
    });
    return eigen_vectors_and_values;
}

vector<Eigen::Matrix2cd> generate_all_zero_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N; i++){
        list_rhoi.push_back((Eigen::Matrix2cd() << 1, 0, 0, 0).finished());
    }
    return list_rhoi;
}

vector<Eigen::Matrix2cd> generate_all_one_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N; i++){
        list_rhoi.push_back((Eigen::Matrix2cd() << 0, 0, 0, 1).finished());
    }
    return list_rhoi;
}

vector<Eigen::Matrix2cd> generate_neel_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N; i++){
        if(i % 2 == 0)
            list_rhoi.push_back((Eigen::Matrix2cd() << 0, 0, 0, 1).finished());
        else
            list_rhoi.push_back((Eigen::Matrix2cd() << 1, 0, 0, 0).finished());
    }
    return list_rhoi;
}

vector<Eigen::Matrix2cd> generate_all_plus_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N; i++){
        list_rhoi.push_back((Eigen::Matrix2cd() << 0.5, 0.5, 0.5, 0.5).finished());
    }
    return list_rhoi;
}

vector<Eigen::Matrix2cd> generate_half_half_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N / 2; i++){
        list_rhoi.push_back((Eigen::Matrix2cd() << 0, 0, 0, 1).finished());
    }
    for(int i = N / 2; i < N; i++){
        list_rhoi.push_back((Eigen::Matrix2cd() << 1, 0, 0, 0).finished());
    }
    return list_rhoi;
}

vector<Eigen::Matrix2cd> generate_random_product_state(){
    vector<Eigen::Matrix2cd> list_rhoi;
    for(int i = 0; i < N; i++){
        vec u = random_unit_vector();
        list_rhoi.push_back(0.5 * PauliI + 0.5 * (x(u) * PauliX + y(u) * PauliY + z(u) * PauliZ));
    }
    return list_rhoi;
}


/*
    Heisenberg evolution
*/

Eigen::MatrixXcd Heisenberg_Evolved_ai_dagger_ai(int i, double t, Eigen::MatrixXd &c_in_eta, Eigen::MatrixXd &eta_in_c, vector<double> &Energy){
    Eigen::MatrixXcd coef_eta(2*N, 2*N);
    coef_eta.setZero();

    for(int l = 0; l < 2 * N; l++){
        for(int k = 0; k < 2 * N; k++){
            complex<double> phase = exp(im * t * ((l < N ? -1 * Energy[l] : Energy[l-N]) + (k < N ? -1 * Energy[k] : Energy[k-N])));
            coef_eta(l, k) = phase * c_in_eta(i+N, l) * c_in_eta(i, k);
        }
    }

    return eta_in_c.transpose() * coef_eta * eta_in_c;
}

Eigen::MatrixXcd precompute_the_Z_parity(vector<Eigen::Matrix2cd> list_rhoi){
    Eigen::MatrixXcd precomputed_Z(N, N);

    for(int pos1 = 0; pos1 < N; pos1 ++){
        precomputed_Z(pos1, pos1) = 1.0;
        if(pos1 < N-1) precomputed_Z(pos1, pos1+1) = 1.0;

        for(int pos2 = pos1+2; pos2 < N; pos2 ++)
            precomputed_Z(pos1, pos2) = precomputed_Z(pos1, pos2-1) * ((-PauliZ * list_rhoi[pos2-1]).trace());
    }

    return precomputed_Z;
}

complex<double> product_state_exp_fermionic_oper(vector<Eigen::Matrix2cd> list_rhoi, int k, int l, Eigen::MatrixXcd precomputed_Z){
    int pos1 = (k < N ? k : k - N);
    int pos2 = (l < N ? l : l - N);
    Eigen::Matrix2cd P_pos1 = (k < N ? Lowering : Raising);
    Eigen::Matrix2cd P_pos2 = (l < N ? Lowering : Raising);

    if(pos1 == pos2){
        P_pos1 = P_pos1 * P_pos2;
        P_pos2 = P_pos1;
    }
    else{
        if(pos1 < pos2) P_pos1 = -P_pos1 * PauliZ;
        if(pos1 > pos2) P_pos2 = -PauliZ * P_pos2;

        if(pos1 > pos2){
            swap(pos1, pos2);
            swap(P_pos1, P_pos2);
        }
    }

    complex<double> val = precomputed_Z(pos1, pos2);

    if(pos1 == pos2){
        val *= (P_pos1 * list_rhoi[pos1]).trace();
    }
    else{
        val *= (P_pos1 * list_rhoi[pos1]).trace();
        val *= (P_pos2 * list_rhoi[pos2]).trace();
    }

    return val;
}

double product_state_exp(vector<Eigen::Matrix2cd> list_rhoi, Eigen::MatrixXcd coef_c, int approx = -1){
    Eigen::MatrixXcd precomputed_Z = precompute_the_Z_parity(list_rhoi);

    complex<double> val = 0.0;
    for(int k = 0; k < 2 * N; k++){
        for(int l = 0; l < 2 * N; l++){
            int pos1 = (k < N ? k : k - N);
            int pos2 = (l < N ? l : l - N);
            if(approx != -1 && abs(pos1 - pos2) > approx) continue;

            val += coef_c(k, l) * product_state_exp_fermionic_oper(list_rhoi, k, l, precomputed_Z);
        }
    }

    assert(abs(val.imag()) < 1e-7);
    return val.real() * 2 - 1;
}

int main(){
    vector<double> field_h;
    // for(int i = 0; i < N; i ++){
    //     field_h.push_back(0.5);
    // }
    cout << "Field: ";
    for(int i = 0; i < N; i ++){
        field_h.push_back(0.5);
    }
    cout << endl;

    Eigen::MatrixXd A(N, N);
    Eigen::MatrixXd B(N, N);

    for(int i = 0; i < N; i++)
        A(i, i) = field_h[i];
    for(int i = 0; i < N-1; i++){
        A(i, i+1) = 0.5;
        A(i+1, i) = 0.5;
    }
    for(int i = 0; i < N-1; i++){
        B(i, i+1) = 0.5 * gamma_param;
        B(i+1, i) = -0.5 * gamma_param;
    }

    Eigen::MatrixXd X = (A - B) * (A + B);
    vector<tuple<double, Eigen::VectorXd>> eigen_vectors_and_values_phi = compute_eig(X);

    vector<double> eigen_values;
    Eigen::MatrixXd Phi(N, N);
    int index = 0;
    for(auto const vect : eigen_vectors_and_values_phi){
        eigen_values.push_back(get<0>(vect));
        Phi.col(index) = get<1>(vect);
        index++;
    }

    X = (A + B) * (A - B);
    vector<tuple<double, Eigen::VectorXd>> eigen_vectors_and_values_psi = compute_eig(X);
    vector<double> eigen_values2;
    Eigen::MatrixXd Psi(N, N);
    index = 0;
    for(auto const vect : eigen_vectors_and_values_psi){
        eigen_values2.push_back(get<0>(vect));
        Psi.col(index) = get<1>(vect);
        index++;
    }

    vector<double> Energy;

    for(int i = 0; i < N; i++){
        Energy.push_back(sqrt(eigen_values[i] + 1e-14));

        if( (Phi.col(i).transpose() * (A - B) - Energy[i] * Psi.col(i).transpose()).norm() > 1e-6 )
            Psi.col(i) = Psi.col(i) * -1;

        assert(abs(eigen_values[i] - eigen_values2[i]) < 1e-6);
        assert((Phi.col(i).transpose() * (A - B) - Energy[i] * Psi.col(i).transpose()).norm() < 1e-6);
    }

    printf("Energy: ");
    for(auto x : Energy){
        printf("%f, ", x);
    }
    printf("\n");

    Eigen::MatrixXd G = (Phi.transpose() + Psi.transpose()) / 2;
    Eigen::MatrixXd H = (Phi.transpose() - Psi.transpose()) / 2;
    Eigen::MatrixXd GH(G.rows(), G.cols()+H.cols());
    GH << G, H; // horizontal concat
    Eigen::MatrixXd HG(G.rows(), G.cols()+H.cols());
    HG << H, G; // horizontal concat
    Eigen::MatrixXd eta_in_c(G.rows()+H.rows(), G.cols()+H.cols());
    eta_in_c << GH,
                HG; // vertical concat
    Eigen::MatrixXd c_in_eta = eta_in_c.inverse();

    double t_ls[8] = {1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0};

    cout << "All0 ";
    cout << "[";
    for(auto t : t_ls){
        double val = product_state_exp(generate_all_zero_state(), Heisenberg_Evolved_ai_dagger_ai(15, t, c_in_eta, eta_in_c, Energy));
        cout << val << ",";
    }
    cout << "]" << endl;

    cout << "All1 ";
    cout << "[";
    for(auto t : t_ls){
        double val = product_state_exp(generate_all_one_state(), Heisenberg_Evolved_ai_dagger_ai(15, t, c_in_eta, eta_in_c, Energy));
        cout << val << ",";
    }
    cout << "]" << endl;

    cout << "++-- ";
    cout << "[";
    for(auto t : t_ls){
        double val = product_state_exp(generate_half_half_state(), Heisenberg_Evolved_ai_dagger_ai(15, t, c_in_eta, eta_in_c, Energy));
        cout << val << ",";
    }
    cout << "]" << endl;

    cout << "Neel ";
    cout << "[";
    for(auto t : t_ls){
        double val = product_state_exp(generate_neel_state(), Heisenberg_Evolved_ai_dagger_ai(15, t, c_in_eta, eta_in_c, Energy));
        cout << val << ",";
    }
    cout << "]" << endl;

    ofstream stateFile("states.txt");
    ofstream valFile("values.txt");

    int num_states = 10000;

    for(int i = 0; i < num_states; i++){
        if(i % 1000 == 0) printf("%d\n", i);

        vector<Eigen::Matrix2cd> state = generate_random_product_state();

        stateFile << "[";
        for(int j = 0; j < N; j++){
            stateFile << (PauliX * state[j]).trace().real() << ",";
            stateFile << (PauliY * state[j]).trace().real() << ",";
            stateFile << (PauliZ * state[j]).trace().real() << ",";
        }
        stateFile << "]" << endl;

        valFile << "[";
        for(auto t : t_ls){
            double val = product_state_exp(state, Heisenberg_Evolved_ai_dagger_ai(15, t, c_in_eta, eta_in_c, Energy));
            valFile << val << ",";
        }
        valFile << "]" << endl;
    }
}
