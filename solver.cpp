#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

/*
 * To open the solver do these steps:
 * 1) build solver with simple "make solver" command in terminal
 * 2) run solver with command "./solver [gamma value] [Jacobi/Richardson/GaussSeidel method]"
 *
 * Replace the first [] brackets with value of gamma parameter
 * Replace the second [] brackets with desired method of solver,
 * accepted values are: Jacobi/jacobi, GaussSeidel/gaussSeidel, Richardson/richardson
 *
 * examples: ./solver 10 Jacobi
 *           ./solver 4/5 gaussSeidel
 *           ./solver 0.8 richardson
 *
 * Solver will then print out the solution vector and number of iterations
 * (if the convergence of the iteration method is fulfilled)
 */

// import most common Eigen types
using namespace Eigen;

class Solver {
    public:
        Solver(double gamma, double precision){
            this->gamma = gamma;
            this->precision = precision;
            A = MatrixXd(size,size);
            L = MatrixXd(size,size);
            U = MatrixXd(size,size);
            D = MatrixXd(size,size);
            b = VectorXd(size);
            initialize();
        }

        void solveJacobi () {
            MatrixXd Q (size, size);
            Q = D;
            solve(Q);
        }

        void solveGaussSeidel (int omega) {
            MatrixXd Q(size, size);
            Q = ((1 / omega) * D) + L;
            solve(Q);
        }

        void solveRichardson () {
            MatrixXd Q (size, size);
            Q.setZero();
            for ( int i = 0; i < size; ++i)Q(i,i) = 1;
            solve(Q);
        }

    private:
        const int size = 20;
        double gamma;
        double precision;
        int iterationsCount = 0;
        MatrixXd A, L, U, D;
        VectorXd b;

        void initialize ( ) {
            //set matrix A according to assignment
            A.setZero();
            for ( int i = 0; i < size; ++i) A(i,i) = gamma;
            for ( int i = 0; i < size - 1; ++i )
            {
                A(i + 1, i ) = -1;
                A(i, i + 1 ) = -1;
            }

            //set vector b according to assignment
            b(0) = gamma - 1;
            b(size - 1) = gamma -1;
            for ( int i =1; i < size - 1; ++i) b(i) = (gamma - 2);

            D.setZero();
            for ( int i = 0; i < size; ++i) D(i,i) = A(i,i);

            //lower triangular part of matrix A
            L = A.triangularView<Lower>();
            for ( int i = 0; i < size; ++i) L(i,i) = 0;

            //upper triangular part of matrix A
            U = A.triangularView<Upper>();
            for ( int i = 0; i < size; ++i) U(i,i) = 0;
        }

        bool isNotPrecise ( VectorXd x)
        {
            //checking if the required precision is achieved
            return (((A * x) - b).norm() / b.norm()) > precision;
        }

        bool isConvergent ( MatrixXd Qinvert ){
            MatrixXd E (size, size);
            E.setZero();
            for ( int i = 0; i < size; ++i) E(i,i) = 1;

            auto eigenvalues = (E - (Qinvert * A)).eigenvalues();
            double maxAbsoluteEigenValue = 0;

            //finding highest absolute eigen value (spectral radius)
            //can converge if the spectral radius is lower than 1
            for (int i = 0; i < eigenvalues.size(); ++i )
                if (std::fabs(eigenvalues(i).real()) > maxAbsoluteEigenValue) maxAbsoluteEigenValue = eigenvalues(i).real();
            if (maxAbsoluteEigenValue < 1.0)  return true;
            return false;
        }

        void solve ( MatrixXd Q) {
            MatrixXd Qinvert(size, size);
            Qinvert = Q.inverse();

            //solution vector
            VectorXd x (size);
            x.setZero();

            if (isConvergent(Qinvert)) {
                while (isNotPrecise(x)) {
                    x = (Qinvert * (Q - A) * x) + Qinvert * b;
                    iterationsCount++;
                }
                std::cout << "Vector x:" << std::endl << x << std::endl << "Iterations: " << iterationsCount << std::endl;
            }
            else std::cout << "Convergence not fulfilled." << std::endl;
        }
};

double parseNumericInput (std::string arg)
{
    int delim = arg.find("/");
    if (delim != std::string::npos)
    {
        return std::stod(arg.substr(0, delim))/std::stod(arg.substr(delim + 1, arg.size()));
    }
    else return std::stod(arg);
}

int main( int argc, char *argv[] ) {
    double gamma = parseNumericInput(std::string(argv[1]));
    const double precision = pow(10, -6);
    int omega = 1; //parameter for Q matrix in Gauss-Seidel

    Solver solver (gamma, precision);

    //depending on the second argument, selected choice of algorithm is either Jacobi or Gauss-Seidel
    if ( std::string(argv[2]) == "Jacobi" || std::string(argv[2]) == "jacobi") solver.solveJacobi();
    else if ( std::string(argv[2]) == "gaussSeidel" || std::string(argv[2]) == "GaussSeidel") solver.solveGaussSeidel(omega);
    else if ( std::string(argv[2]) == "Richardson" || std::string(argv[2]) == "richardson") solver.solveRichardson();
    else std::cout << "Wrong method input, try again.";

    return 0;
}
