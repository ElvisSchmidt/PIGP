#include <iostream>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <boost/numeric/odeint.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <fstream>
using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;

// simple linspace
VectorXd linspace(double start, double end, int num){
    VectorXd result = VectorXd::Zero(num);
    double step = (end - start)/(num-1);
    for(int i=0;i<num;i++) result[i] = start+i*step;

    return result;
}

typedef Eigen::Matrix<double, Dynamic, 1> state_type;
// --- Function to read initial condition from CSV ---
state_type read_initial_condition(const std::string &filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(fin, line); // skip header

    std::vector<double> y0_vec;
    while (std::getline(fin, line)) {
        size_t comma = line.find(',');
        // double x = std::stod(line.substr(0, comma)); // optional: x values
        double y = std::stod(line.substr(comma + 1));
        y0_vec.push_back(y);
    }

    state_type y0(y0_vec.size());
    for (size_t i = 0; i < y0_vec.size(); ++i) {
        y0(i) = y0_vec[i];
    }

    return y0;
}
int main() {
    // Grid and discretization
    //cout << "Here"<<endl;
state_type y0 = read_initial_condition("Initial_interpolated_data.csv");
// Initial condition
int m = y0.size();

    double left = 0, right = 1900, h = (left-right)/(m-1);
    const double D = 0.03;
    const double K = 1.86;
    const double lambda = 0.04;
    // Grid points
    VectorXd x = linspace(left, right, m);

    // Difference operator D2 (second derivative)
  MatrixXd D2 = MatrixXd::Zero(m,m);
    for(int i=1;i<m-1;i++){
    D2(i,i-1) = 1.0;
    D2(i,i) = -2.0;
    D2(i,i+1) = 1.0;
    }
D2(0,0) = 1.0;
D2(0,1) = -2.0;
D2(0,2) = 1.0;
D2(m-1,m-3) = 1.0;
D2(m-1,m-2) = -2.0;
D2(m-1,m-1) = 1.0;
    D2 /= (h*h);

    // Mass / norm matrix H 
    MatrixXd H = MatrixXd::Identity(m,m);
    H(0,0) = 0.5; H(m-1,m-1) = 0.5;
    H *= h;
    MatrixXd H_inv = H.inverse();
    // Boundary vectors
    VectorXd e_1 = VectorXd::Zero(m); e_1[0] = 1;
    VectorXd e_m = VectorXd::Zero(m); e_m[m-1] = 1;

    // No flux boundary conditions
    MatrixXd L(2,m);
    L.row(0).setZero(); //Forward difference
    L(0,0) = -3.0/(2*h); 
    L(0,1) = 4.0/(2*h);
    L(0,2) = -1.0/(2*h);
    
    L.row(1).setZero(); //Backward difference
    L(1,m-3) = 1.0/(2*h);
    L(1,m-2) = -4.0/(2*h);
    L(1,m-1) = 3.0/(2*h); 
    

    // Projection operator P = I - H^-1 L^T (L H^-1 L^T)^-1 L
    MatrixXd Id = MatrixXd::Identity(m,m);
    MatrixXd inv_mx = (L*H_inv*L.transpose()).inverse();
    MatrixXd P = Id - H_inv * L.transpose() * inv_mx * L;

    // Final operator: projected first derivative
    VectorXd Id_vec = VectorXd::Ones(m);
    MatrixXd Final_Operator = D2 * P;
    typedef Eigen::Matrix<double,Dynamic,1> state_type;
    // ODE system lambda
    auto system = [&m,&Final_Operator, &P,&K,&D,&lambda]( state_type &y, state_type &dy_dt, double /*t*/){
        y.resize(m);
        dy_dt.resize(m);
        MatrixXd Id = MatrixXd::Identity(m,m);
        VectorXd RHS = y.array()*(1.0-y.array()/K);
        dy_dt = P*(D*Final_Operator*y + lambda* RHS);
};


    // Solver: RK4 with integrate_const



    auto observer = [](const VectorXd &y, double t){
        // Uncomment to see evolution
        //cout << t << " " << y.transpose() << endl;
    };
    //Initial condtiotions
    //cout << "Here"<<endl;
    typedef runge_kutta4<state_type> stepper_type;
    stepper_type rk4_stepper;
    state_type y = 1000* y0;
    double t = 0.0;
    double t_end = 48.1;
    double dt = 0.03;

vector<state_type> solutions;
vector<double> times;


while(t <= t_end) {
    solutions.push_back(y);  // store current state
    times.push_back(t);      // optional: store time
    rk4_stepper.do_step(system, y, t, dt);
    t += dt;
}


    std::ofstream fout("solutionKPP.csv");

    // Header
    fout << "x,t,y\n";

    for (size_t k = 0; k < solutions.size(); ++k) {
        double ti = times[k];
        for (int i = 0; i < x.size(); ++i) {
            double xi = x(i);
            double yi = solutions[k](i);
            fout << xi << "," << ti << "," << yi << "\n";
        }
    }

fout.close();
}