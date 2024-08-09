// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;

// typedef Eigen::SparseMatrix<double> SparseMatrix;


// get the 0-based knot index
// [[Rcpp::export]]
int knotindex(double x,Eigen::VectorXd t) {
  int q = t.size();
  int k=0;
  if (x < t(0)) return -1;
  while(x>=t(k)){
    k++;
    if (k >= q) break;
  }

  return k-1;
}


// B-splines
double weight(double x,Eigen::VectorXd t,int i,int k) {
  if (t(i+k-1) != t(i-1))
    return((x - t(i-1))/(t(i+k-1)-t(i-1)));
  return 0.;
}
// [[Rcpp::export]]
double Bspline(double x,int j,Eigen::VectorXd t,int p) {
  // Evaluate the jth B-spline
  // B_p(x) of order p (degree p-1) at x
  if (p==1)
    return(x>=t(j-1) && x<t(j+1-1));

  double w1 = weight(x,t,j,p-1);
  double w2 = weight(x,t,j+1,p-1);
  double b1 = Bspline(x,j,t,p-1);
  double b2 = Bspline(x,j+1,t,p-1);

  return w1*b1 + (1.-w2)*b2;
}
// [[Rcpp::export]]
Eigen::VectorXd Bsplinevec(double x,Eigen::VectorXd t,int p) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  int k = knotindex(x,t);
  for (int i=(k-(p-1));i<k+1;i++)
    b(i) = Bspline(x,i+1,t,p);
  return b;
}
// [[Rcpp::export]]
Eigen::VectorXd BsplinevecCon(double x,Eigen::VectorXd t,int p,Eigen::MatrixXd Z) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  int k = knotindex(x,t);
  for (int i=(k-(p-1));i<k+1;i++)
    b(i) = Bspline(x,i+1,t,p);
  return b.transpose()*Z;
}
// [[Rcpp::export]]
double BsplinevecConJ(double x,Eigen::VectorXd t,int p,Eigen::MatrixXd Z, int j) {
  if (j == 1)
    return 1;
  Eigen::VectorXd b = BsplinevecCon(x, t, p, Z);
  // j starts from 1, in order to match the structure in R.
  return b(j-2);
}

// [[Rcpp::export]]
Eigen::VectorXd Bsplinevec2(double x,Eigen::VectorXd t,int p) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  for (int i=0;i<m;i++)
    b(i) = Bspline(x,i+1,t,p);
  return b;
}
// [[Rcpp::export]]
Eigen::VectorXd Bsplinevec2Con(double x,Eigen::VectorXd t,int p,Eigen::MatrixXd Z) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  for (int i=0;i<m;i++)
    b(i) = Bspline(x,i+1,t,p);
  return b.transpose()*Z;
}
// [[Rcpp::export]]
double Bsplinevec2ConJ(double x,Eigen::VectorXd t,int p,Eigen::MatrixXd Z, int j) {
  if (j == 1)
    return 1;
  Eigen::VectorXd b = Bsplinevec2Con(x, t, p, Z);
  // j starts from 1, in order to match the structure in R.
  return b(j-2);
}


// [[Rcpp::export]]
void BsplinevecFill(double x,Eigen::VectorXd t,int p, int col, Eigen::SparseMatrix<double>& Gx) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  int k = knotindex(x,t);
  for (int i=(k-(p-1));i<k+1;i++)
    Gx.insert(i, col) = Bspline(x,i+1,t,p);
}

// [[Rcpp::export]]
void Bsplinevec2Fill(double x,Eigen::VectorXd t,int p, int col, Eigen::SparseMatrix<double>& Gx) {
  int m = t.size() - p;
  Eigen::VectorXd b(m);
  b.setZero();
  for (int i=0;i<m;i++)
    Gx.insert(i, col) = Bspline(x,i+1,t,p);
}


// [[Rcpp::export]]
List Integral(Eigen::VectorXd knots_x, Eigen::VectorXd knots_w, int kx, int kw,
              double maxL, Eigen::MatrixXd Zx, Eigen::MatrixXd Zw, Eigen::VectorXd t,
              Eigen::VectorXd alphax, bool OnlyAlphaxD) {
  int Nt = t.size();
  double tt;
  Eigen::MatrixXd AlphaxD(Nt, kw); // output
  // AlphaxD.setZero();
  Eigen::MatrixXd D(kx, kw);
  double Dij = 0.;
  double l0;
  double l1;
  double space;
  // Eigen::VectorXd leven;
  int knotsl0;
  int knotsl1;


  Eigen::VectorXd knots_w_int = knots_w.segment(3, knots_w.size() - 6);
  int kw_int = knots_w_int.size();
  knots_w_int(0) = 0;
  knots_w_int(kw_int-1) = maxL;

  Eigen::VectorXd knots_x_int = knots_x.segment(3, knots_x.size() - 6);
  int kx_int = knots_x_int.size();
  knots_x_int(0) = 0;
  knots_x_int(kx_int-1) = t(Nt-1);

  Eigen::VectorXd knots_x_I;

  Eigen::MatrixXd Gx(kx, 4);
  Eigen::MatrixXd Gw(kw, 4);

  Eigen::VectorXd tmpx(kx);
  Eigen::VectorXd tmpw(kw);
  Eigen::MatrixXd P(4, 4);
  // Eigen::PartialPivLU<Eigen::MatrixXd> luP(P); // Perform LU decomposition for solving linear system.
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      P(row, col) = std::pow(-1. + 2. * row / 3.0, col);
    }
  }
  Eigen::MatrixXd Pinv = P.inverse();
  Eigen::MatrixXd Pl(4,4);

  Eigen::MatrixXd H(4, 4); // Matrix H
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      H(row, col) = (1. + std::pow(-1., row + col)) / (row + col + 1);
    }
  }
  Eigen::MatrixXd W = Pinv.transpose() * H * Pinv;


  // get coefficient and g (evaluated spline function) for x at different knots
  std::vector<Eigen::MatrixXd> CoefXlist;
  std::vector<Eigen::MatrixXd> Gxlist;
  std::vector<Eigen::MatrixXd> Dxlist;
  double x0;
  double x1;
  for (int xx = 0; xx < (kx_int-1); xx++){
    Gx.setZero();
    x0 = knots_x_int(xx);
    x1 = knots_x_int(xx+1);
    space = x1-x0;
    // set Pl
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        Pl(row, col) = std::pow(x0 + row*space/3.0, col);
      }
    }
    for (int s = 0; s < 4; s++) {
      if(x0 + s*space/3.0 < knots_x(4)) {
        // index issue at start
        tmpx << 1.0, Bsplinevec2Con(x0 + s*space/3.0, knots_x, 4, Zx);
      } else {
        tmpx << 1.0, BsplinevecCon(x0 + s*space/3.0, knots_x, 4, Zx);
      }
      Gx.col(s) = tmpx;
    }
    Gxlist.push_back(Gx); // * Pl.inverse().transpose()
    CoefXlist.push_back(Gx * Pl.inverse().transpose());
    Dxlist.push_back( (space/2) * Gx * W * Gx.transpose() );
  }

  // get g (evaluated spline function) for w at different knots
  Eigen::MatrixXd CoefWtmp;
  std::vector<Eigen::MatrixXd> Gwlist;
  std::vector<Eigen::MatrixXd> GwWGwlist;
  for (int ll = 0; ll < (kw_int-1); ll++){
    Gw.setZero();
    l0 = knots_w_int(ll);
    l1 = knots_w_int(ll+1);
    space = l1-l0;
    // // set Pl
    // for (int row = 0; row < 4; row++) {
    //   for (int col = 0; col < 4; col++) {
    //     Pl(row, col) = std::pow(l0 + row*space/3.0, col);
    //   }
    // }
    for (int s = 0; s < 4; s++) {
      if(l0 + s*space/3.0 < knots_w(4)) {
         // index issue at start
        tmpx << 1.0, Bsplinevec2Con(l0 + s*space/3.0, knots_w, 4, Zw);
      } else {
        tmpx << 1.0, BsplinevecCon(l0 + s*space/3.0, knots_w, 4, Zw);
      }
      Gw.col(s) = tmpx;
    }
    Gwlist.push_back(Gw); // * Pl.inverse().transpose()
    GwWGwlist.push_back(Gw * W * Gw.transpose());
  }


  // Integral 1: Obtain alpha_x * D
  for (int ts = 0; ts < Nt; ts++) {
    tt = t(ts);
    // std::cout << tt << std::endl;

    D.setZero(); // initialize D
    for (int ll = 0; ll < (kw_int-1); ll++){
      l0 = knots_w_int(ll);
      l1 = knots_w_int(ll+1);
      knotsl0 = knotindex(tt-l0, knots_x);
      knotsl1 = knotindex(tt-l1, knots_x);

      if(knotsl1 == knotsl0){
        //[l0, l1]
        space = l1-l0;
        // Gx = Pl*(the common coeff)
        for (int row = 0; row < 4; row++) {
          for (int col = 0; col < 4; col++) {
            Pl(row, col) = std::pow(tt - (l0 + row*space/3.0), col);
          }
        }
        Gx = CoefXlist.at(knotsl0-3) * Pl.transpose();
        // Gw
        Gw = (space/2.0)*Gwlist.at(ll);
        D += Gx * W * Gw.transpose();

      } else {
        // get coefficient for wl in [l0, l1]
        space = l1-l0;
        for (int row = 0; row < 4; row++) { // reset Pl
          for (int col = 0; col < 4; col++) {
            Pl(row, col) = std::pow(l0 + row*space/3.0, col);
          }
        }
        CoefWtmp = Gwlist.at(ll) * Pl.inverse().transpose();

        knots_x_I.resize(knotsl0-knotsl1+2);
        knots_x_I << tt-l1, knots_x.segment(knotsl1+1, knotsl0-knotsl1), tt-l0;
        for (int i = 0; i < (knotsl0-knotsl1+1); i++){
          if((i == 0 || (i+1) == (knotsl0-knotsl1+1))) {
            // the starting and ending segment
            x0 = knots_x_I(i);
            x1 = knots_x_I(i+1);
            space = x1-x0;
            for (int row = 0; row < 4; row++) {
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(x0 + row*space/3.0, col);
              }
            }
            Gx = CoefXlist.at(knotindex(x0, knots_x)-3) * Pl.transpose();
            // Gw
            for (int row = 0; row < 4; row++) { // reset Pl
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
              }
            }
            Gw = (space/2.0) * CoefWtmp * Pl.transpose();

            D += Gx * W * Gw.transpose();
          } else {
            x0 = knots_x_I(i);
            x1 = knots_x_I(i+1);
            space = x1-x0;
            Gx = Gxlist.at(knotindex(x0, knots_x)-3);

            // Gw
            for (int row = 0; row < 4; row++) { // reset Pl
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
              }
            }
            Gw = (space/2.0) * CoefWtmp * Pl.transpose();

            D += Gx * W * Gw.transpose();
          }

        }
      }
    }
    AlphaxD.row(ts) = alphax.transpose() * D;
  }

  Eigen::MatrixXd Dw(kw, kw);
  Eigen::VectorXd Xt2(Nt);
  Eigen::MatrixXd D2(kx, kx);
  if(OnlyAlphaxD){
    // Error when kw < kx here: free(): invalid size. TODO: fix it!
    return List::create(Named("AlphaxD") = AlphaxD);
  }else{
    // Integral 2: Obtain Dw for constraint 1: \int w(l)^2 dl = 1
    // equivalent to w^T %*% Dw %*% w = 1
    // Dw_{ij} = \int b_i(l) b_j(l) dl
    // indeed, same as the matrix Cw in paper.
    Dw.setZero();
    for (int ll = 0; ll < (kw_int-1); ll++){
      l0 = knots_w_int(ll);
      l1 = knots_w_int(ll+1);
      space = l1-l0;
      // Gw = Gwlist.at(ll);
      // Dw += ((space/2.0) * Gw) * W * Gw.transpose();
      Dw += (space/2.0) * GwWGwlist.at(ll);
    }

    // Integral 3: sqrt(\int_{l in [0,maxL]} X(t-l)^2 dl)
    // For upper bound of E = max_t sqrt(\int X(t-l)^2 dl)
    l0 = 0;
    l1 = maxL;
    for (int ts = 0; ts < Nt; ts++) {
      tt = t(ts);
      D2.setZero(); // initialize D2
      knotsl0 = knotindex(tt-l0, knots_x);
      knotsl1 = knotindex(tt-l1, knots_x);
      knots_x_I.resize(knotsl0-knotsl1+2);
      knots_x_I << tt-l1, knots_x.segment(knotsl1+1, knotsl0-knotsl1), tt-l0;
      for (int i = 0; i < (knotsl0-knotsl1+1); i++){
        if((i == 0 || (i+1) == (knotsl0-knotsl1+1))) {
          // the starting and ending segment
          x0 = knots_x_I(i);
          x1 = knots_x_I(i+1);
          space = x1-x0;
          for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
              Pl(row, col) = std::pow(x0 + row*space/3.0, col);
            }
          }
          Gx = CoefXlist.at(knotindex(x0, knots_x)-3) * Pl.transpose();
          D2 += ((space/2.0) * Gx) * W * Gx.transpose();
        } else {
          x0 = knots_x_I(i);
          D2 += Dxlist.at(knotindex(x0, knots_x)-3);
        }

      }
      Xt2(ts) = sqrt(alphax.transpose() * D2 * alphax);
    }
    return List::create(Named("AlphaxD") = AlphaxD,
                        Named("Dw") = Dw,
                        Named("Xt2") = Xt2);

  }
}


// // [[Rcpp::export]]
// List Integral_interpolate_OLD(Eigen::VectorXd knots_x, Eigen::VectorXd knots_w, int kx, int kw,
//               double maxL, Eigen::MatrixXd Zw, Eigen::VectorXd t,
//               Eigen::VectorXd alphax, bool OnlyAlphaxD) {
//   int Nt = t.size();
//   double tt;
//   Eigen::MatrixXd AlphaxD(Nt, kw); // output
//   // AlphaxD.setZero();
//   Eigen::MatrixXd D(kx, kw);
//   double l0;
//   double l1;
//   double space;
//   // Eigen::VectorXd leven;
//   int knotsl0;
//   int knotsl1;
//   int knotsx0;


//   Eigen::VectorXd knots_w_int = knots_w.segment(3, knots_w.size() - 6);
//   int kw_int = knots_w_int.size();
//   knots_w_int(0) = 0;
//   knots_w_int(kw_int-1) = maxL;

//   Eigen::VectorXd knots_x_int = knots_x.segment(3, knots_x.size() - 6);
//   int kx_int = knots_x_int.size();
//   knots_x_int(0) = 0;
//   knots_x_int(kx_int-1) = std::max(t(Nt-1), knots_x_int(kx_int-2));

//   // std::cout << knots_x_int << std::endl;

//   Eigen::VectorXd knots_x_I;
//   Eigen::MatrixXd Gx(kx, 4);
//   Eigen::SparseMatrix<double> GxSparse(kx, 4);
//   Eigen::MatrixXd Gw(kw, 4);

//   Eigen::VectorXd tmpx(kx);
//   Eigen::VectorXd tmpw(kw);
//   Eigen::MatrixXd P(4, 4);
//   // Eigen::PartialPivLU<Eigen::MatrixXd> luP(P); // Perform LU decomposition for solving linear system.
//   for (int row = 0; row < 4; row++) {
//     for (int col = 0; col < 4; col++) {
//       P(row, col) = std::pow(-1. + 2. * row / 3.0, col);
//     }
//   }
//   Eigen::MatrixXd Pinv = P.inverse();
//   Eigen::MatrixXd Pl(4,4);
//   Eigen::MatrixXd Pltmp(4,4);

//   Eigen::MatrixXd H(4, 4); // Matrix H
//   for (int row = 0; row < 4; row++) {
//     for (int col = 0; col < 4; col++) {
//       H(row, col) = (1. + std::pow(-1., row + col)) / (row + col + 1);
//     }
//   }
//   Eigen::MatrixXd W = Pinv.transpose() * H * Pinv;

//   std::cout << "Start construction for x... " << std::endl;
//   // get coefficient and g (evaluated spline function) for x at different knots
//   std::vector<Eigen::MatrixXd> CoefXlist;
//   std::vector<Eigen::SparseMatrix<double>> Gxlist;
//   std::vector<Eigen::MatrixXd> Plxlist;
//   double x0;
//   double x1;
//   for (int xx = 0; xx < (kx_int-1); xx++){
//     GxSparse.setZero();
//     x0 = knots_x_int(xx);
//     x1 = knots_x_int(xx+1);
//     space = x1-x0;
//     // set Pl
//     for (int row = 0; row < 4; row++) {
//       for (int col = 0; col < 4; col++) {
//         Pl(row, col) = std::pow(x0 + row*space/3.0, col);
//       }
//     }

//     for (int s = 0; s < 4; s++) {
//       if(x0 + s*space/3.0 < knots_x(4)) {
//         // index issue at start
//         Bsplinevec2Fill(x0 + s*space/3.0, knots_x, 4, s, GxSparse);
//       } else {
//         BsplinevecFill(x0 + s*space/3.0, knots_x, 4, s, GxSparse);
//       }
//     }
//     GxSparse.makeCompressed();


//     Gxlist.push_back(GxSparse); // * Pl.inverse().transpose()
//     Plxlist.push_back(Pl.inverse().transpose());
//   }
//   std::cout << "Finish construction for x. " << std::endl;
//   std::cout << "Start construction for w... " << std::endl;
//   // get g (evaluated spline function) for w at different knots
//   Eigen::MatrixXd CoefWtmp;
//   std::vector<Eigen::MatrixXd> Gwlist;
//   std::vector<Eigen::MatrixXd> GwWGwlist;
//   for (int ll = 0; ll < (kw_int-1); ll++){
//     Gw.setZero();
//     l0 = knots_w_int(ll);
//     l1 = knots_w_int(ll+1);
//     space = l1-l0;

//     for (int s = 0; s < 4; s++) {
//       if(l0 + s*space/3.0 < knots_w(4)) {
//          // index issue at start
//         tmpw << 1.0, Bsplinevec2Con(l0 + s*space/3.0, knots_w, 4, Zw);
//       } else {
//         tmpw << 1.0, BsplinevecCon(l0 + s*space/3.0, knots_w, 4, Zw);
//       }
//       Gw.col(s) = tmpw;
//     }
//     Gwlist.push_back(Gw); // * Pl.inverse().transpose()
//     GwWGwlist.push_back(Gw * W * Gw.transpose());
//   }

//   std::cout << "Finish construction for w." << std::endl;

//   std::cout << "Start Obtain alpha_x * D ... " << std::endl;
//   // Integral 1: Obtain alpha_x * D
//   for (int ts = 0; ts < Nt; ts++) {
//     tt = t(ts);
//     // std::cout << tt << std::endl;

//     D.setZero(); // initialize D
//     for (int ll = 0; ll < (kw_int-1); ll++){
//       l0 = knots_w_int(ll);
//       l1 = knots_w_int(ll+1);
//       knotsl0 = knotindex(tt-l0, knots_x);
//       knotsl1 = knotindex(tt-l1, knots_x);

//       if(knotsl1 == knotsl0){
//         //[l0, l1]
//         space = l1-l0;
//         // Gx = Pl*(the common coeff)
//         for (int row = 0; row < 4; row++) {
//           for (int col = 0; col < 4; col++) {
//             Pl(row, col) = std::pow(tt - (l0 + row*space/3.0), col);
//           }
//         }
//         Pltmp = Plxlist.at(knotsl0-3) * Pl.transpose();
//         // Gw
//         Gw = (space/2.0)*Gwlist.at(ll);
//         D += Gxlist.at(knotsl0-3) * (Pltmp * W * Gw.transpose());
//       } else {
//         // get coefficient for wl in [l0, l1]
//         space = l1-l0;
//         for (int row = 0; row < 4; row++) { // reset Pl
//           for (int col = 0; col < 4; col++) {
//             Pl(row, col) = std::pow(l0 + row*space/3.0, col);
//           }
//         }
//         CoefWtmp = Gwlist.at(ll) * Pl.inverse().transpose();

//         knots_x_I.resize(knotsl0-knotsl1+2);
//         knots_x_I << tt-l1, knots_x.segment(knotsl1+1, knotsl0-knotsl1), tt-l0;
//         for (int i = 0; i < (knotsl0-knotsl1+1); i++){
//           if((i == 0 || (i+1) == (knotsl0-knotsl1+1))) {
//             // the starting and ending segment
//             x0 = knots_x_I(i);
//             x1 = knots_x_I(i+1);
//             space = x1-x0;
//             for (int row = 0; row < 4; row++) {
//               for (int col = 0; col < 4; col++) {
//                 Pl(row, col) = std::pow(x0 + row*space/3.0, col);
//               }
//             }
//             knotsx0 = knotindex(x0, knots_x);
//             Pltmp = Plxlist.at(knotsx0-3) * Pl.transpose();
//             // Gw
//             for (int row = 0; row < 4; row++) { // reset Pl
//               for (int col = 0; col < 4; col++) {
//                 Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
//               }
//             }
//             Gw = (space/2.0) * CoefWtmp * Pl.transpose();

//             D += Gxlist.at(knotsx0-3) * (Pltmp * W * Gw.transpose());
//           } else {
//             x0 = knots_x_I(i);
//             x1 = knots_x_I(i+1);
//             space = x1-x0;

//             // Gw
//             for (int row = 0; row < 4; row++) { // reset Pl
//               for (int col = 0; col < 4; col++) {
//                 Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
//               }
//             }
//             Gw = (space/2.0) * CoefWtmp * Pl.transpose();

//             knotsx0 = knotindex(x0, knots_x);
//             D += Gxlist.at(knotsx0-3) * (W * Gw.transpose());
//           }

//         }
//       }
//     }
//     AlphaxD.row(ts) = alphax.transpose() * D;
//   }

//   std::cout << "Finish Obtain alpha_x * D ... " << std::endl;

//   Eigen::MatrixXd Dw(kw, kw);
//   if(OnlyAlphaxD){
//     return List::create(Named("AlphaxD") = AlphaxD);
//   }else{
//     // Integral 2: Obtain Dw for constraint 1: \int w(l)^2 dl = 1
//     // equivalent to w^T %*% Dw %*% w = 1
//     // Dw_{ij} = \int b_i(l) b_j(l) dl
//     // indeed, same as the matrix Cw in paper.
//     std::cout << "Start Obtain Dw ... " << std::endl;
//     Dw.setZero();
//     for (int ll = 0; ll < (kw_int-1); ll++){
//       l0 = knots_w_int(ll);
//       l1 = knots_w_int(ll+1);
//       space = l1-l0;
//       // Gw = Gwlist.at(ll);
//       // Dw += ((space/2.0) * Gw) * W * Gw.transpose();
//       Dw += (space/2.0) * GwWGwlist.at(ll);
//     }
//     std::cout << "Finish Obtain Dw." << std::endl;

//     return List::create(Named("AlphaxD") = AlphaxD,
//                         Named("Dw") = Dw);
//   }
// }



// deBoor's algorithm for spline. From https://github.com/awstringer1/semibmd/blob/main/src/helpers.cpp
// [[Rcpp::export]]
double deBoor(double x,Eigen::VectorXd t,Eigen::VectorXd beta,int p) {
  // x: evaluation point
  // t: knots
  // beta: coefficients
  // p: spline ORDER (cubic = 4)
  //
  // outputs f(x) via deBoor's algorithm
  int k = knotindex(x,t);
  Eigen::VectorXd d(p);
  for (int j=0;j<p;j++) {
    // if (j+k-(p-1) < 0 || j+k-(p-1) > beta.size()-1) {
	  //   Rcout << "deBoor with x = " << x << ", p=" << p << ", beta length=" << beta.size() << ", k=" << k << ", t length=" << t.size() <<  std::endl;
	  //   Rcout << "Accessing index " << j+k-(p-1) << "of beta" << std::endl;
    // }
    d(j) = beta(j+k-(p-1));
  }

  double a =0.;
  for (int r=1;r<p;r++)
    for (int j=(p-1);j>r-1;j--) {
      if (j+k-(p-1) < 0 || j+k-(p-1) > t.size()-1 || j+1+k-r < 0 || j+1+k-r > t.size()-1) {
        Rcout << "x: " << x << std::endl;
        Rcout << "Accessing indices " << j+k-(p-1) << ", " << j+1+k-r << " of t" << std::endl;
      }
      a = (x-t(j+k-(p-1))) / (t(j+1+k-r) - t(j+k-(p-1)));
      d(j) = (1. - a)*d(j-1) + a*d(j);
    }

    return d(p-1);
}


// TODO: check deboor at the boundary.
// TODO: better way of construction for w.
// [[Rcpp::export]]
List Integral_interpolate(Eigen::VectorXd knots_x, Eigen::VectorXd knots_w, int kx, int kw,
                          double maxL, Eigen::MatrixXd Zw, Eigen::VectorXd t,
                          Eigen::VectorXd alphax, bool OnlyAlphaxD) {
  int Nt = t.size();
  double tt;

  Eigen::MatrixXd AlphaxD(Nt, kw); // output
  Eigen::VectorXd AlphaxDi(kw);


  double Xt2i = 0.0;
  double l0;
  double l1;
  double space;
  // Eigen::VectorXd leven;
  int knotsl0;
  int knotsl1;
  int knotsx0;

  Eigen::VectorXd knots_w_int = knots_w.segment(3, knots_w.size() - 6);
  int kw_int = knots_w_int.size();
  knots_w_int(0) = 0;
  knots_w_int(kw_int-1) = maxL;

  Eigen::VectorXd knots_x_int = knots_x.segment(3, knots_x.size() - 6);
  int kx_int = knots_x_int.size();
  knots_x_int(0) = 0;
  knots_x_int(kx_int-1) = std::max(t(Nt-1), knots_x_int(kx_int-2));

  // std::cout << knots_x_int << std::endl;

  Eigen::VectorXd knots_x_I;
  Eigen::VectorXd Gx(4);
  Eigen::MatrixXd Gw(kw, 4);

  Eigen::VectorXd tmpw(kw);
  Eigen::MatrixXd P(4, 4);
  // Eigen::PartialPivLU<Eigen::MatrixXd> luP(P); // Perform LU decomposition for solving linear system.
  // obtain the P matrix
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      P(row, col) = std::pow(-1. + 2. * row / 3.0, col);
    }
  }
  Eigen::MatrixXd Pl(4,4);
  Eigen::MatrixXd Pltmp(4,4);
  // inv(P) for solving polynomial coefficients
  Eigen::MatrixXd Pinv = P.inverse();
  

  Eigen::MatrixXd H(4, 4); // Matrix H in the middle
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      H(row, col) = (1. + std::pow(-1., row + col)) / (row + col + 1);
    }
  }
  Eigen::MatrixXd W = Pinv.transpose() * H * Pinv; // Matrix W in the middle

  // std::cout << "Start construction for x... " << std::endl;
  // get coefficient and g (evaluated spline function) for x at different knots
  Eigen::VectorXd DxVec(kx_int);
  Eigen::MatrixXd CoefxMat(4, kx_int);
  Eigen::MatrixXd GxMat(4, kx_int);
  double x0;
  double x1;
  for (int xx = 0; xx < (kx_int-1); xx++){
    x0 = knots_x_int(xx);
    x1 = knots_x_int(xx+1);
    space = x1-x0;
    // set Pl
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        Pl(row, col) = std::pow(x0 + row*space/3.0, col);
      }
    }

    for (int s = 0; s < 4; s++) {
      // if(x0 + s*space/3.0 < knots_x(4)) {
      //   // index issue at start
      //   Bsplinevec2Fill(x0 + s*space/3.0, knots_x, 4, s, GxSparse);

      // } else {
      //   BsplinevecFill(x0 + s*space/3.0, knots_x, 4, s, GxSparse);
      // }
      Gx(s) = deBoor(x0 + s*space/3.0, knots_x, alphax, 4);
    }

    CoefxMat.col(xx) = Pl.inverse() * Gx; // coefficient for X within X's knots
    GxMat.col(xx) = Gx; // evaluate X within X's knots
    DxVec(xx) = (space/2) * Gx.transpose() * W * Gx; 
  }

  // std::cout << "Finish construction for x. " << std::endl;

  // std::cout << "Start construction for w... " << std::endl;
  // get g (evaluated spline function) for w at different knots
  Eigen::MatrixXd CoefWtmp;
  std::vector<Eigen::MatrixXd> Gwlist;
  std::vector<Eigen::MatrixXd> GwWGwlist;
  for (int ll = 0; ll < (kw_int-1); ll++){
    Gw.setZero();
    l0 = knots_w_int(ll);
    l1 = knots_w_int(ll+1);
    space = l1-l0;

    for (int s = 0; s < 4; s++) {
      if(l0 + s*space/3.0 < knots_w(4)) {
         // index issue at start
        tmpw << 1.0, Bsplinevec2Con(l0 + s*space/3.0, knots_w, 4, Zw);
      } else {
        tmpw << 1.0, BsplinevecCon(l0 + s*space/3.0, knots_w, 4, Zw);
      }
      Gw.col(s) = tmpw;
    }
    Gwlist.push_back(Gw); // * Pl.inverse().transpose() // evaluate w within w's knots
    GwWGwlist.push_back(Gw * W * Gw.transpose());
  }

  // std::cout << "Finish construction for w." << std::endl;

  // std::cout << "Start Obtain alpha_x * D ... " << std::endl;
  // Integral 1: Obtain alpha_x * D
  for (int ts = 0; ts < Nt; ts++) {
    tt = t(ts);
    // std::cout << tt << std::endl;
    AlphaxDi.setZero();
    for (int ll = 0; ll < (kw_int-1); ll++){
      l0 = knots_w_int(ll);
      l1 = knots_w_int(ll+1);
      knotsl0 = knotindex(tt-l0, knots_x);
      knotsl1 = knotindex(tt-l1, knots_x);

      if(knotsl1 == knotsl0){
        //[l0, l1]
        space = l1-l0;
        // Gx = Pl*(the common coeff)
        for (int row = 0; row < 4; row++) {
          for (int col = 0; col < 4; col++) {
            Pl(row, col) = std::pow(tt - (l0 + row*space/3.0), col);
          }
        }
        Gx = Pl * CoefxMat.col(knotsl0-3);
        // Gw
        Gw = (space/2.0)*Gwlist.at(ll);
        AlphaxDi += Gx.transpose() * W * Gw.transpose();
      } else {
        // get coefficient for wl in [l0, l1]
        space = l1-l0;
        for (int row = 0; row < 4; row++) { // reset Pl
          for (int col = 0; col < 4; col++) {
            Pl(row, col) = std::pow(l0 + row*space/3.0, col);
          }
        }
        CoefWtmp = Gwlist.at(ll) * Pl.inverse().transpose();

        knots_x_I.resize(knotsl0-knotsl1+2);
        knots_x_I << tt-l1, knots_x.segment(knotsl1+1, knotsl0-knotsl1), tt-l0;
        for (int i = 0; i < (knotsl0-knotsl1+1); i++){
          if((i == 0 || (i+1) == (knotsl0-knotsl1+1))) {
            // the starting and ending segment
            x0 = knots_x_I(i);
            x1 = knots_x_I(i+1);
            space = x1-x0;
            for (int row = 0; row < 4; row++) {
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(x0 + row*space/3.0, col);
              }
            }
            Gx = Pl * CoefxMat.col(knotindex(x0, knots_x)-3);
            // Gw
            for (int row = 0; row < 4; row++) { // reset Pl
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
              }
            }
            Gw = (space/2.0) * CoefWtmp * Pl.transpose();

            AlphaxDi += Gx.transpose() * W * Gw.transpose();
          } else {
            x0 = knots_x_I(i);
            x1 = knots_x_I(i+1);
            space = x1-x0;
            Gx = GxMat.col(knotindex(x0, knots_x)-3);

            // Gw
            for (int row = 0; row < 4; row++) { // reset Pl
              for (int col = 0; col < 4; col++) {
                Pl(row, col) = std::pow(tt-x0 - row*space/3.0, col);
              }
            }
            Gw = (space/2.0) * CoefWtmp * Pl.transpose();

            AlphaxDi += Gx.transpose() * W * Gw.transpose();
          }

        }
      }
    }
    AlphaxD.row(ts) = AlphaxDi;
  }

  // std::cout << "Finish Obtain alpha_x * D ... " << std::endl;
  Eigen::MatrixXd Dw(kw, kw);
  if(OnlyAlphaxD){
    return List::create(Named("AlphaxD") = AlphaxD);
  }else{
    // Integral 2: Obtain Dw for constraint 1: \int w(l)^2 dl = 1
    // equivalent to w^T %*% Dw %*% w = 1
    // Dw_{ij} = \int b_i(l) b_j(l) dl
    // indeed, same as the matrix Cw in paper.
    Dw.setZero();
    for (int ll = 0; ll < (kw_int-1); ll++){
      l0 = knots_w_int(ll);
      l1 = knots_w_int(ll+1);
      space = l1-l0;
      // Gw = Gwlist.at(ll);
      // Dw += ((space/2.0) * Gw) * W * Gw.transpose();
      Dw += (space/2.0) * GwWGwlist.at(ll);
    }
    Eigen::VectorXd Xt2(Nt);
    Xt2.setZero();
    // Integral 3: sqrt(\int_{l in [0,maxL]} X(t-l)^2 dl)
    // For upper bound of E = max_t sqrt(\int X(t-l)^2 dl)
    // std::cout << "Start Obtain sqrt(int_{l in [0,maxL]} X(t-l)^2 dl) ... " << std::endl;
    l0 = 0;
    l1 = maxL;
    for (int ts = 0; ts < Nt; ts++) {
      tt = t(ts);
      Xt2i = 0.0; // initialize Xt2
      knotsl0 = knotindex(tt-l0, knots_x);
      knotsl1 = knotindex(tt-l1, knots_x);
      knots_x_I.resize(knotsl0-knotsl1+2);
      knots_x_I << tt-l1, knots_x.segment(knotsl1+1, knotsl0-knotsl1), tt-l0;
      for (int i = 0; i < (knotsl0-knotsl1+1); i++){
        if((i == 0 || (i+1) == (knotsl0-knotsl1+1))) {
          // the starting and ending segment
          x0 = knots_x_I(i);
          x1 = knots_x_I(i+1);
          space = x1-x0;
          for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
              Pl(row, col) = std::pow(x0 + row*space/3.0, col);
            }
          }
          knotsx0 = knotindex(x0, knots_x);
          Gx = Pl * CoefxMat.col(knotsx0-3);
          Xt2i += ((space/2.0) * Gx.transpose()) * W * Gx;
        } else {
          x0 = knots_x_I(i);
          Xt2i += DxVec(knotindex(x0, knots_x)-3);
        }
      }
      Xt2(ts) = sqrt(Xt2i);
    }
    // std::cout << "Finish Obtain sqrt(int_{l in [0,maxL]} X(t-l)^2 dl)" << std::endl;

    return List::create(Named("AlphaxD") = AlphaxD,
                        Named("Dw") = Dw,
                        Named("Xt2") = Xt2);
  }

}


// // [[Rcpp::export]]
// Eigen::VectorXd Interpolate(Eigen::MatrixXd X, Eigen::MatrixXd Zx, Eigen::VectorXd y) {
//   Eigen::MatrixXd Xcon = X * Zx;
//   Eigen::MatrixXd Xrepa(Xcon.rows(), Xcon.cols() + 1);
//   Xrepa << Eigen::MatrixXd::Ones(Xrepa.rows(), 1), Xcon;
//   // TODO: use banded chol decomp
//   Eigen::MatrixXd XrepatXrepa = Xrepa.transpose() * Xrepa;
//   Eigen::VectorXd beta = XrepatXrepa.ldlt().solve(Xrepa.transpose() * y);

//   return beta;
//   // return List::create(Named("beta") = beta);
// }



// [[Rcpp::export]]
Eigen::VectorXd Interpolate(Eigen::SparseMatrix<double> X, Eigen::VectorXd y) {

  Eigen::SparseMatrix<double> XtX = X.transpose() * X;
  // Eigen::VectorXd beta = XtX.ldlt().solve(X.transpose() * y);
  Eigen::VectorXd Xty = X.transpose() * y;
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(XtX);
  Eigen::VectorXd beta = chol.solve(Xty);

  return beta;
}
