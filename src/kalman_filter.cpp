#include "kalman_filter.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
    Eigen::MatrixXd &H_in, Eigen::MatrixXd &Hj_in, Eigen::MatrixXd &R_in,
    Eigen::MatrixXd &R_ekf_in, Eigen::MatrixXd &Q_in);  {
      x_ = x_in;
      P_ = P_in;
      F_ = F_in;
      H_ = H_in;
      Hj_ = Hj_in;
      R_ = R_in;
      R_ekf_ = R_ekf_in;
      Q_ = Q_in;
      I_ = Eigen::MatrixXd::Identity(4,4);
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_*P_*Ft+Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = P_*Ht*S_i;

  x_ = x_ + K*y;
  P_ = (I_-K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float pos_x_2 = pow(x_[0],2);
  float pos_y_2 = pow(x_[1],2);

  float ro = pow(pos_x_2+pos_y_2,0.5);
  float theta = atan2(x_[0],x_[1]);
  float ro_dot = (x_[0]*x_[2]+x_[1]*x_[3])/ro;

  Hj_ = tools.CalculateJacobian( x_ );

  VectorXd h_prime = VectorXd(3);
  h_prime << ro, theta, ro_dot;
  VectorXd y = z - h_prime;
  while (y[1]>M_PI) {
    y[1]-2*M_PI;
  }
  while (y[1]<(-M_PI)) {
    y[1]+2*M_PI;
  }

  MatrixXd Ht = Hj_.transpose();
  MatrixXd S = Hj_*P_*Ht + R_ekf_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = P_*Ht*S_i;

  x_ = x_ + K*y;
  P_ = (I_-K*Hj_)*P_;
}
