#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <cstring>
#include <cmath>
#include <armadillo>
#include "mpi.h"
using std::cout;
using std::endl;

static const double PI   =  3.1415926535897932;

double RandExp(const double width);
double rtnorm(double mean, double tau, double sd);
double rho_to_alpha(double rho, double scale);
double logpdf_beta(double x, double a, double b);
struct Opts 
{
  int num_burn;
  int num_thin;
  int num_save;
  int num_print;
  bool update_sigma_mu;
  bool update_s;
  bool update_alpha;
  bool update_beta;
  bool update_gamma;
  bool update_tau;
  bool update_tau_mean;
};

struct rho_loglik {
  double mean_log_s;
  double p;
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  double operator() (double rho) {

    double alpha = rho_to_alpha(rho, alpha_scale);

    double loglik = alpha * mean_log_s
      + lgamma(alpha)
      - p * lgamma(alpha / p)
      + logpdf_beta(rho, alpha_shape_1, alpha_shape_2);
    return loglik;

  }
};


double activation(double x, double c, double tau) ;
double growth_prior(int node_depth, double gamma, double beta);
double ldcauchy(double x,double loc,double sig);
double cauchy_jacobian(double tau, double sigma_hat);

int sample_class(const arma::vec& probs) ;
int sample_class(int n);
double logit(double x);
double expit(double x);
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision);
arma::mat choll(const arma::mat& Sigma);
double log_sum_exp(const arma::vec& x);
double rlgam(double shape) ;
double ldexp(double x,double rate);
double logprior_tau(double tau, double tau_rate);
double log_tau_trans(double tau_new) ;
bool do_mh(double loglik_new, double loglik_old,double new_to_old, double old_to_new) ;
double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old,double temperature );


struct Node;

struct Hypers {
  double alpha;
  double beta;
  double gamma;
  double sigma;
  double sigma_mu;
  double shape;
  int np;
  double width;
  double tau_rate;
  double binaryOffset;
  int binary;
  double temperature;
  int num_tree;
  int num_groups;
  double sigma_hat;
  double sigma_mu_hat;
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;
  double totalcount;  
  arma::vec s;
  arma::vec logs;
  arma::uvec group;
  arma::vec rho_propose;
  std::vector<std::vector<unsigned int> > group_to_vars;

  void UpdateSigma(const arma::vec& r,const double myrank);
  void UpdateSigmaMu(const arma::vec& means);
  void UpdateAlpha();
  void UpdateGamma(std::vector<Node*>& forest);
  void UpdateBeta(std::vector<Node*>& forest);
  void UpdateTauRate(const std::vector<Node*>& forest);
  int SampleVar() const;

};


struct Node {
  bool is_leaf;
  bool is_root;
  Node* left;
  Node* right;
  Node* parent;
  // Branch parameters
  int var;
  double val;
  double lower;
  double upper;
  double tau;
  // Leaf parameters
  double mu;
  // Data for computing weights
  double current_weight;
  unsigned int nid;
  // Functions
  Node * getptr(const unsigned int Tid);
  void Root(const Hypers& hypers);
  void GetLimits();
  double Getval(const double MyVar);
  void AddLeaves();
  void BirthLeaves(const Hypers& hypers);
  bool is_left();
  void GetW(const arma::mat& X, int i);
  void DeleteLeaves();
  void UpdateMu(const arma::vec& Y, const arma::mat& X, const Hypers& hypers,const double myrank);
  void UpdateMuA(double * ML,const double myrank); 
  void UpdateMuB(double * ML,const double myrank);   
  void UpdateTau(const arma::vec& Y, const arma::mat& X, const Hypers& hypers,const double myrank);
  bool UpdateTauA(const arma::vec& Y, const arma::mat& X, const Hypers& hypers,const double myrank,double OLogLT,double * Mulist,int MCount);  
  bool UpdateTauB(const arma::vec& Y, const arma::mat& X, const Hypers& hypers,const double myrank,double OLogLT,double * Mulist,int MCount);  
  void SetTau(double tau_new);
  void SetOneTau(double tau_new);  
//  double loglik_tau(double tau_new, const arma::mat& X, const arma::vec& Y, const Hypers& hypers,const double myrank);
  Node();
  ~Node();

};

void UpdateWTM(Node* N, int & index,const arma::mat& X,const arma::vec W, const double t,arma::mat & wtm);
void UpdateS(std::vector<Node*>& forest, Hypers& hypers);
double growth_prior(int leaf_depth, const Hypers& hypers);
void InitHypers(Hypers & THypers ) ;
void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) ;
arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers) ;
std::vector<Node*> init_forest(const arma::mat& X, const arma::vec& Y,const Hypers& hypers) ;
void GetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank);
void GetSuffStatsB(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out);
void EGetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank);
double tree_loglik(Node* node, int node_depth, double gamma, double beta);
double forest_loglik(std::vector<Node*>& forest, double gamma, double beta);
arma::vec get_tau_vec(const std::vector<Node*>& forest) ;
int depth(Node* node);
arma::vec predict(Node* n, const arma::mat& X, const Hypers& hypers) ;
arma::vec predict(const std::vector<Node*>& forest,const arma::mat& X,const Hypers& hypers);
Node* birth_node(Node* tree, double* leaf_node_probability) ;
double LogLT(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank);
double LogLTA(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank,int & Mucount,double * Mulist );
double LogLTB(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank,int & Mucount,double * Mulist );
double probability_node_birth(Node* tree);
std::vector<Node*> not_grand_branches(Node* tree);
void not_grand_branches(std::vector<Node*>& ngb, Node* node); 
void leaves(Node* x, std::vector<Node*>& leafs) ;
std::vector<Node*> leaves(Node* x);
void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,const Hypers& hypers,const double myrank);
Node* death_node(Node* tree, double* p_not_grand);
void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,const Hypers& hypers,const double myrank);

void change_decision_rule(Node* tree, const arma::mat& X, const arma::vec& Y, const Hypers& hypers,const double myrank);
void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank);
void TreeBackfitOne(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank);
void TreeBackfit_Par(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank);
void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat, Hypers& hypers, const arma::mat& X, const arma::vec& Y, const Opts& opts,const double myrank) ;
arma::vec get_means(std::vector<Node*>& forest);
void get_means(Node* node, std::vector<double>& means);
Node* rand(std::vector<Node*> ngb) ;
arma::vec loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers);









double lbeta(double a,double b);

double RandExp(const double width)
{
    return (-1*log(arma::randu())*width) ;
}

double rtnorm(double mean, double tau, double sd)
{
  double x, z, lambda;

  /* Christian Robert's way */
  //assert(mean < tau); //assert unnecessary: Rodney's way
  tau = (tau - mean)/sd;

  /* originally, the function did not draw this case */
  /* added case: Rodney's way */
  if(tau<=0.) {
    /* draw until we get one in the right half-plane */
    do { z=arma::randn(); } while (z < tau);
  }
  else {
    /* optimal exponential rate parameter */
    lambda = 0.5*(tau + sqrt(tau*tau + 4.0));

    /* do the rejection sampling */
    do {
      z = RandExp(1.0)/lambda + tau;
      //z = lambda*gen.exp() + tau;
    } while (arma::randu() > exp(-0.5*pow(z - lambda, 2.)));
  }

  /* put x back on the right scale */
  x = z*sd + mean;

  //assert(x > 0); //assert unnecessary: Rodney's way
  return(x);

}


double lbeta(double a,double b)
{
  return lgamma(a)+lgamma(b)-lgamma(a+b);
}


double logpdf_beta(double x, double a, double b) {
  double result;
  result= (a-1.0) * log(x) + (b-1.0) * log(1 - x) - lgamma(a) -lgamma(b)+lgamma(a+b) ;
  return result;
}

double activation(double x, double c, double tau) {
  return 1.0 - expit((x - c) / tau);
}


double growth_prior(int node_depth, double gamma, double beta) {
  return gamma * pow(1.0 + node_depth, -beta);
}


double ldcauchy(double x,double loc,double sig)
{
  return log(sig/PI/(sig*sig+(x-loc)*(x-loc)));
}

double cauchy_jacobian(double tau, double sigma_hat) 
{
  double sigma = pow(tau, -0.5);
  double out = ldcauchy(sigma, 0.0, sigma_hat);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);
  return out;
}

double rho_to_alpha(double rho, double scale) 
{
  return scale * rho / (1.0 - rho);
}

int sample_class(const arma::vec& probs) 
{
  double U = arma::randu();
  double foo = 0.0;
  int K = probs.size();
  for(int k = 0; k < K; k++) 
  {
    foo += probs(k);
      if(U < foo) {    return(k);    }
  }
  return K - 1;
}


int sample_class(int n) 
{
  double U = arma::randu();
  double p = 1.0 / ((double)n);
  double foo = 0.0;

  for(int k = 0; k < n; k++) 
  {
    foo += p;
    if(U < foo) {
      return k;
    }
  }
  return n - 1;
}

double logit(double x) 
{
  return log(x) - log(1.0-x);
}

double expit(double x) 
{
  return 1.0 / (1.0 + exp(-x));
}

arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision) 
{
  arma::vec z = arma::zeros<arma::vec>(mean.size());
  z.randn();
  arma::mat Sigma = inv_sympd(Precision);
  arma::mat L = chol(Sigma, "lower");
  arma::vec h = mean + L * z;
  return h;
}

arma::mat choll(const arma::mat& Sigma) 
{
  return chol(Sigma);
}

double log_sum_exp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}


double rlgam(double shape) 
{ arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape,1.0));
  if(shape >= 0.1) return log(v(0));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = arma::randu();
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(arma::randu()) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(arma::randu())) break;
  } while(true);
  return -z/a;
}

double tau_proposal(double tau) 
{
  double U = 2.0 * arma::randu() - 1;
  return pow(5.0, U) * tau;
}

double ldexp(double x,double rate)
{
   return (log(rate)-rate*x) ;
}


double logprior_tau(double tau, double tau_rate) 
{
  return ldexp(tau,  tau_rate);
}

double log_tau_trans(double tau_new) 
{
  return -log(tau_new);
}

bool do_mh(double loglik_new, double loglik_old,double new_to_old, double old_to_new) 
{
  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;
  return log(arma::randu()) < cutoff ? true : false;
}

double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old,double temperature) {

  double SSE = dot(r,r) * temperature;
  double n = r.size() * temperature;

  double shape = 0.5 * n + 1.0;
  double scale = 2.0 / SSE;
  arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape,scale));
  double sigma_prop = pow(v(0), -0.5);

  double tau_prop = pow(sigma_prop, -2.0);
  double tau_old = pow(sigma_old, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
    cauchy_jacobian(tau_old, sigma_hat);

  return log(arma::randu()) < loglik_rat ? sigma_prop : sigma_old;

}

void Hypers::UpdateSigma(const arma::vec& r,const double myrank) {

  double *SSEN = new double[2];
  double *SSEND = new double[2];
  
  SSEN[0] = dot(r,r) * temperature;
  SSEN[1] = r.size() * temperature;
  SSEND[0]=0;
  SSEND[1]=0;
  

	MPI_Reduce(SSEN,SSEND,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);


  if (myrank==0)
  {  
  double shape = 0.5 * SSEND[1] + 1.0;
  double scale = 2.0 / SSEND[0];
  arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape,scale));
  double sigma_prop = pow(v(0), -0.5);

  double tau_prop = pow(sigma_prop, -2.0);
  double tau_old = pow(sigma, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) - cauchy_jacobian(tau_old, sigma_hat);

  sigma = (log(arma::randu()) < loglik_rat ? sigma_prop : sigma);
  }

	delete[] SSEN;
	delete[] SSEND;
}

void Hypers::UpdateSigmaMu(const arma::vec& means) {
  sigma_mu = update_sigma(means, sigma_mu_hat, sigma_mu,1.0);
}

void Hypers::UpdateAlpha() {
  arma::vec logliks = arma::zeros<arma::vec>(rho_propose.size());

  rho_loglik loglik;
  loglik.mean_log_s = mean(logs);
  loglik.p = (double)s.size();
  loglik.alpha_scale = alpha_scale;
  loglik.alpha_shape_1 = alpha_shape_1;
  loglik.alpha_shape_2 = alpha_shape_2;

  for(unsigned int i = 0; i < rho_propose.size(); i++) {
    logliks(i) = loglik(rho_propose(i));
  }

  logliks = exp(logliks - log_sum_exp(logliks));
  double rho_up = rho_propose(sample_class(logliks));
  alpha = rho_to_alpha(rho_up, alpha_scale);

}

void Hypers::UpdateGamma(std::vector<Node*>& forest) {
  double loglik = forest_loglik(forest, gamma, beta);

  for(int i = 0; i < 10; i++) {
    double gamma_prop = 0.5 * arma::randu() + 0.5;
    double loglik_prop = forest_loglik(forest, gamma_prop, beta);
    if(log(arma::randu()) < loglik_prop - loglik) {
      gamma = gamma_prop;
      loglik = loglik_prop;
    }
  }
}

void Hypers::UpdateBeta(std::vector<Node*>& forest) {

  double loglik = forest_loglik(forest, gamma, beta);

  for(int i = 0; i < 10; i++) {
    double beta_prop = fabs(arma::randn() *2.0);
    double loglik_prop = forest_loglik(forest, gamma, beta_prop);
    if(log(arma::randu()) < loglik_prop - loglik) {
      beta = beta_prop;
      loglik = loglik_prop;
    }
  }
}

void Hypers::UpdateTauRate(const std::vector<Node*>& forest) {

  arma::vec tau_vec = get_tau_vec(forest);
  double shape_up = forest.size() + 1.0;
  double rate_up = arma::sum(tau_vec) + 0.1;
  double scale_up = 1.0 / rate_up;
  arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape_up, scale_up));

  tau_rate = v(0);

}


int Hypers::SampleVar() const {

  int group_idx = sample_class(s);
  int var_idx = sample_class(group_to_vars[group_idx].size());

  return group_to_vars[group_idx][var_idx];
}

void Node::Root(const Hypers& hypers) 
{
  is_leaf = true;
  is_root = true;
  left = this;
  right = this;
  parent = this;

  nid=1;
  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;
  tau = hypers.width;

  mu = 0.0;
  current_weight = 1.0;
}

void Node::GetLimits() 
{
  Node* y = this;
  lower = 0.0;
  upper = 1.0;
  bool my_bool = y->is_root ? false : true;
  while(my_bool) {
    bool is_left = y->is_left();
    y = y->parent;
    my_bool = y->is_root ? false : true;
    if(y->var == var) {
      my_bool = false;
      if(is_left) {
        upper = y->val;
        lower = y->lower;
      }
      else {
        upper = y->upper;
        lower = y->val;
      }
    }
  }
}


double Node::Getval(const double MyVar) 
{ 
  double Mylower;
  double Myupper;
  Node* y = this;
  Mylower = 0.0;
  Myupper = 1.0;
  bool my_bool = y->is_root ? false : true;
  while(my_bool) {
    bool is_left = y->is_left();
    y = y->parent;
    my_bool = y->is_root ? false : true;
    if(y->var == MyVar) {
      my_bool = false;
      if(is_left) {
        Myupper = y->val;
        Mylower = y->lower;
      }
      else {
        Myupper = y->upper;
        Mylower = y->val;
      }
    }
  }
  return  ( (Myupper - Mylower) * arma::randu() + Mylower ) ;
}





Node * Node::getptr(const unsigned int Tid)
{
  if(this->nid ==Tid ) return this;
  if(is_leaf) return 0;
  Node * lp=left->getptr(Tid);
  if(lp) return lp;
  Node * rp=right->getptr(Tid);
  if(rp) return rp;
  return 0;
}


void Node::AddLeaves() {
  left = new Node;
  right = new Node;
  is_leaf = false;

  left->is_leaf = true;
  left->parent = this;
  left->right = left;
  left->left = left;
  left->var = 0;
  left->val = 0.0;
  left->is_root = false;
  left->lower = 0.0;
  left->upper = 1.0;
  left->mu = 0.0;
  left->current_weight = 0.0;
  left->tau = tau;
  left->nid=nid*2;

  right->is_leaf = true;
  right->parent = this;
  right->right = right;
  right->left = right;
  right->var = 0;
  right->val = 0.0;
  right->is_root = false;
  right->lower = 0.0;
  right->upper = 1.0;
  right->mu = 0.0;
  right->current_weight = 0.0;
  right->tau = tau;
  right->nid=nid*2+1;

}

void Node::BirthLeaves(const Hypers& hypers) {
  if(is_leaf) {
    AddLeaves();
    var = hypers.SampleVar();
    GetLimits();
    val = (upper - lower) * arma::randu() + lower;
  }
}

bool Node::is_left() {
  return (this == this->parent->left);
}

void Node::GetW(const arma::mat& X, int i) 
{
  if(!is_leaf) 
  {

    double weight = activation(X(i,var), val, tau);
    left->current_weight = weight * current_weight;
    right->current_weight = (1 - weight) * current_weight;
    left->GetW(X,i);
    right->GetW(X,i);

  }
}

void Node::DeleteLeaves() 
{
  delete left;
  delete right;
  left = this;
  right = this;
  is_leaf = true;
}

void Node::UpdateMu(const arma::vec& Y, const arma::mat& X, const Hypers& hypers,const double myrank) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  
  arma::vec mu_samp= arma::zeros<arma::vec>(num_leaves);
  double * mu_MeM=mu_samp.memptr();

  arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
  arma::mat Omega_inv =  arma::zeros<arma::mat>(num_leaves, num_leaves);
  GetSuffStats(this, Y, X, hypers, mu_hat, Omega_inv,myrank);
  if (myrank==0)
  {

    mu_samp = rmvnorm(mu_hat, Omega_inv);
    mu_MeM=mu_samp.memptr();

    for(int i = 0; i < num_leaves; i++) {
      leafs[i]->mu = mu_samp(i);
    }
  }
  MPI_Bcast(mu_MeM,num_leaves,MPI_DOUBLE,0,MPI_COMM_WORLD);

  if (myrank!=0)
  {
    for(int i = 0; i < num_leaves; i++) {
      leafs[i]->mu = mu_MeM[i];
    }
  }   
}


void Node::UpdateMuA(double * ML,const double myrank) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  double * MM=new double [num_leaves];
  
  if (myrank==0)
  {
    for(int i = 0; i < num_leaves; i++) {
      leafs[i]->mu = ML[i];
      MM[i]= ML[i];
    }
  }

  MPI_Bcast(MM,num_leaves,MPI_DOUBLE,0,MPI_COMM_WORLD);

  if (myrank!=0)
  {
    for(int i = 0; i < num_leaves; i++) {
      leafs[i]->mu = MM[i];
    }
  }   
  delete [] MM;
}

void Node::UpdateMuB(double * ML,const double myrank) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();

  

    for(int i = 0; i < num_leaves; i++) {
      leafs[i]->mu = ML[i];
    }
}



void Node::UpdateTau(const arma::vec& Y,const arma::mat& X,const Hypers& hypers,const double myrank) 
{

  double tau_old = tau;
  double tau_new ;

  int AType=0;

  double loglik_old = LogLT(this, Y, X, hypers,myrank)+ logprior_tau(tau_old, hypers.tau_rate);

  if (myrank==0)
  {
    tau_new = tau_proposal(tau);
  }
  MPI_Bcast(&tau_new,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  SetTau(tau_new); 

  


  double loglik_new = LogLT(this, Y, X, hypers,myrank) + logprior_tau(tau_new, hypers.tau_rate);




  if (myrank==0)
  { 
    double new_to_old = log_tau_trans(tau_old);
    double old_to_new = log_tau_trans(tau_new);
    bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);

    if(accept_mh) 
    { 
      AType=0;
    }
    else 
    { 
      AType=1;
    }
  }

  MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
  if (AType)
  {
    SetTau(tau_old);
  }

}


bool Node::UpdateTauA(const arma::vec& Y,const arma::mat& X,const Hypers& hypers,const double myrank,double OLogLT,double * Mulist,int Count)  
{

  double tau_old = tau;
  double tau_new ;


  int AType=0;
  int MCount=0;


  double * Tlist =new double [Count];
  double loglik_old = OLogLT+ logprior_tau(tau_old, hypers.tau_rate);

  if (myrank==0)
  {
//    cout<<Log1<<" "<<OLogLT<<"\n"<<std::endl;
    tau_new = tau_proposal(tau);
  }
  MPI_Bcast(&tau_new,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  SetTau(tau_new); 

  


  double loglik_new = LogLTA(this, Y, X, hypers,myrank,MCount,Tlist) + logprior_tau(tau_new, hypers.tau_rate);




  if (myrank==0)
  { 
    double new_to_old = log_tau_trans(tau_old);
    double old_to_new = log_tau_trans(tau_new);
    bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);

    if(accept_mh) 
    { 
      AType=0;
      for (int i=0;i<MCount;i++)
      {
        Mulist[i]=Tlist[i];
      }
    }
    else 
    { 
      AType=1;
    }
  }

  MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
  if (AType)
  {
    SetTau(tau_old);
  }

  delete [] Tlist;
  return !(AType);


}


bool Node::UpdateTauB(const arma::vec& Y,const arma::mat& X,const Hypers& hypers,const double myrank,double OLogLT,double * Mulist,int Count)  
{

  double tau_old = tau;
  double tau_new ;


  int AType=0;
  int MCount=0;


  double * Tlist =new double [Count];
  double loglik_old = OLogLT+ logprior_tau(tau_old, hypers.tau_rate);

  tau_new = tau_proposal(tau);
  SetTau(tau_new); 

  


  double loglik_new = LogLTB(this, Y, X, hypers,myrank,MCount,Tlist) + logprior_tau(tau_new, hypers.tau_rate);





    double new_to_old = log_tau_trans(tau_old);
    double old_to_new = log_tau_trans(tau_new);
    bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);

    if(accept_mh) 
    { 
      AType=0;
      for (int i=0;i<MCount;i++)
      {
        Mulist[i]=Tlist[i];
      }
    }
    else 
    { 
      AType=1;
    }



  if (AType)
  {
    SetTau(tau_old);
  }

  delete [] Tlist;
  return !(AType);


}


void Node::SetTau(double tau_new) {
  tau = tau_new;
  if(!is_leaf) {
    left->SetTau(tau_new);
    right->SetTau(tau_new);
  }
}

void Node::SetOneTau(double tau_new) {
  tau = tau_new;
}

// double Node::loglik_tau(double tau_new, const arma::mat& X,const arma::vec& Y, const Hypers& hypers,const double myrank) {

//   double tau_old = tau;
//   SetTau(tau_new);
//   double out = LogLT(this, Y, X, hypers,myrank);
//   SetTau(tau_old);
//   return out;

// }

Node::Node() {
  is_leaf = true;
  is_root = true;
  left = NULL;
  right = NULL;
  parent = NULL;

  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;
  tau = 1.0;
  mu = 0.0;
  current_weight = 0.0;
}

Node::~Node() {
  if(!is_leaf) {
    delete left;
    delete right;
  }
}

void UpdateS(std::vector<Node*>& forest, Hypers& hypers) 
{
  arma::vec shape_up = hypers.alpha / ((double)hypers.s.size()) * arma::ones<arma::vec>(hypers.s.size());
  shape_up = shape_up + get_var_counts(forest, hypers);

  for(unsigned int i = 0; i < shape_up.size(); i++) 
  {
    hypers.logs(i) = rlgam(shape_up(i));
 //   cout<<hypers.logs(i)<<"\n";
  }
 
  hypers.logs = hypers.logs - log_sum_exp(hypers.logs);
  hypers.s = exp(hypers.logs);
  
}

double growth_prior(int leaf_depth, const Hypers& hypers) 
{
  return hypers.gamma * pow(1.0 + leaf_depth, -hypers.beta);
}

void InitHypers(Hypers & THypers ) 
{ 
  int GRID_SIZE = 1000;
  THypers.num_groups = THypers.group.max() + 1;
  THypers.s = arma::ones<arma::vec>(THypers.num_groups) / ((double)(THypers.num_groups));
  THypers.logs = log(THypers.s);
  THypers.sigma = THypers.sigma_hat;
  THypers.sigma_mu_hat = THypers.sigma_mu;
  THypers.group_to_vars.resize(THypers.s.size());
  for(unsigned int i = 0; i < THypers.s.size(); i++) {
    THypers.group_to_vars[i].resize(0);
  }
  int P = THypers.group.size();
  for(int p = 0; p < P; p++) {
    int idx = THypers.group(p);
    THypers.group_to_vars[idx].push_back(p);
  }
  THypers.rho_propose = arma::zeros<arma::vec>(GRID_SIZE - 1);
  for(int i = 0; i < GRID_SIZE - 1; i++) 
  {
    THypers.rho_propose(i) = (double)(i+1) / (double)(GRID_SIZE);
  }
}

void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) 
{
  if(!node->is_leaf) {
    int group_idx = hypers.group(node->var);
    counts(group_idx) = counts(group_idx) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
}

arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers) 
{
  arma::uvec counts = arma::zeros<arma::uvec>(hypers.s.size());
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}


void UpdateWTM(Node* N, int & index,const arma::mat& X,const arma::vec W, const double t,arma::mat & wtm) 
{
  if(N->is_leaf) 
  {
  
    wtm.col(index)=W ;
    index++;
  }
  else {
    arma::vec C=1.0-1.0/(1.0+exp(-1*(X.col(N->var)-N->val)/t));
    UpdateWTM(N->left, index, X,C%W,t,wtm) ;
    UpdateWTM(N->right, index, X,(1-C)%W,t,wtm) ;
  }
}


std::vector<Node*> init_forest(const arma::mat& X, const arma::vec& Y,const Hypers& hypers) 
{

  std::vector<Node*> forest(0);
  for(int t = 0; t < hypers.num_tree; t++) {
    Node* n = new Node;
    n->Root(hypers);
    forest.push_back(n);
  }
  return forest;
}

void GetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank) {


  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
  arma::vec mu_hat_S = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_S = arma::zeros<arma::mat>(num_leaves, num_leaves);
  arma::vec mu_hat_H = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_H = arma::zeros<arma::mat>(num_leaves, num_leaves);


  if (num_leaves==1)
  {
    mu_hat_S[0]=arma::sum(y);
    Lambda_S(0,0)=X.n_rows;
  }
  else
  {
    for(unsigned int i = 0; i < X.n_rows; i++) {
      n->GetW(X, i);
      for(int j = 0; j < num_leaves; j++) {
        w_i(j) = leafs[j]->current_weight;
      }
      mu_hat_S = mu_hat_S + y(i) * w_i;
      Lambda_S = Lambda_S + w_i * arma::trans(w_i);
    }
  }

  double* mu_hat_S_mem = mu_hat_S.memptr();
  double* Lambda_S_mem = Lambda_S.memptr();

  double* mu_hat_H_mem = mu_hat_H.memptr();
  double* Lambda_H_mem = Lambda_H.memptr();


  MPI_Reduce(mu_hat_S_mem, mu_hat_H_mem, num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(Lambda_S_mem, Lambda_H_mem, num_leaves*num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myrank==0)
  {
    Lambda_H = Lambda_H / pow(hypers.sigma, 2) * hypers.temperature;
    mu_hat_H = mu_hat_H / pow(hypers.sigma, 2) * hypers.temperature;

    Omega_inv_out = Lambda_H + arma::eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
    mu_hat_out = solve(Omega_inv_out, mu_hat_H);

  }  

}


void GetSuffStatsB(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out) {


  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
  arma::vec mu_hat_S = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_S = arma::zeros<arma::mat>(num_leaves, num_leaves);



  if (num_leaves==1)
  {
    mu_hat_S[0]=arma::sum(y);
    Lambda_S(0,0)=X.n_rows;
  }
  else
  {
    for(unsigned int i = 0; i < X.n_rows; i++) {
      n->GetW(X, i);
      for(int j = 0; j < num_leaves; j++) {
        w_i(j) = leafs[j]->current_weight;
      }
      mu_hat_S = mu_hat_S + y(i) * w_i;
      Lambda_S = Lambda_S + w_i * arma::trans(w_i);
    }
  }
    Lambda_S = Lambda_S / pow(hypers.sigma, 2) * hypers.temperature;
    mu_hat_S = mu_hat_S / pow(hypers.sigma, 2) * hypers.temperature;
    Omega_inv_out = Lambda_S + arma::eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
    mu_hat_out = solve(Omega_inv_out, mu_hat_S);
}


void EGetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, const Hypers& hypers,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank) {


  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  int N =y.size();
  int Windex=0;

  arma::vec mu_hat_S = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_S = arma::zeros<arma::mat>(num_leaves, num_leaves);
  arma::vec mu_hat_H = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_H = arma::zeros<arma::mat>(num_leaves, num_leaves);
  arma::mat W_ALL = arma::zeros<arma::mat>(N, num_leaves);
  arma::vec W_ones = arma::ones<arma::vec>(N);



  UpdateWTM(n, Windex,X,W_ones, n->tau,W_ALL);

  //for(unsigned int i = 0; i < X.n_rows; i++) {
  //  n->GetW(X, i);
  //  for(int j = 0; j < num_leaves; j++) {
  //    w_i(j) = leafs[j]->current_weight;
  //  }
  //  mu_hat_S = mu_hat_S + y(i) * w_i;
  //  Lambda_S = Lambda_S + w_i * arma::trans(w_i);
  //}


  mu_hat_S=W_ALL.t() * y;
  Lambda_S=W_ALL.t() * W_ALL;

  double* mu_hat_S_mem = mu_hat_S.memptr();
  double* Lambda_S_mem = Lambda_S.memptr();

  double* mu_hat_H_mem = mu_hat_H.memptr();
  double* Lambda_H_mem = Lambda_H.memptr();


  MPI_Reduce(mu_hat_S_mem, mu_hat_H_mem, num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(Lambda_S_mem, Lambda_H_mem, num_leaves*num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myrank==0)
  {
    Lambda_H = Lambda_H / pow(hypers.sigma, 2) * hypers.temperature;
    mu_hat_H = mu_hat_H / pow(hypers.sigma, 2) * hypers.temperature;

    Omega_inv_out = Lambda_H + arma::eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
    mu_hat_out = solve(Omega_inv_out, mu_hat_H);

  }  

}


double tree_loglik(Node* node, int node_depth, double gamma, double beta) {
  double out = 0.0;
  if(node->is_leaf) {
    out += log(1.0 - growth_prior(node_depth, gamma, beta));
  }
  else {
    out += log(growth_prior(node_depth, gamma, beta));
    out += tree_loglik(node->left, node_depth + 1, gamma, beta);
    out += tree_loglik(node->right, node_depth + 1, gamma, beta);
  }
  return out;
}

double forest_loglik(std::vector<Node*>& forest, double gamma, double beta) {
  double out = 0.0;
  for(unsigned int t = 0; t < forest.size(); t++) {
    out += tree_loglik(forest[t], 0, gamma, beta);
  }
  return out;
}

arma::vec get_tau_vec(const std::vector<Node*>& forest) {
  int t = forest.size();
  arma::vec out = arma::zeros<arma::vec>(t);
  for(int i = 0; i < t; i++) {
    out(i) = forest[i]->tau;
  }
  return out;
}

int depth(Node* node) {
  return node->is_root ? 0 : 1 + depth(node->parent);
}

arma::vec predict(Node* n, const arma::mat& X, const Hypers& hypers) 
{

  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  int N = X.n_rows;
  arma::vec out = arma::zeros<arma::vec>(N);

  for(int i = 0; i < N; i++) {
    n->GetW(X,i);
    for(int j = 0; j < num_leaves; j++) {
      out(i) = out(i) + leafs[j]->current_weight * leafs[j]->mu;
    }
  }
  return out;
}

arma::vec predict(const std::vector<Node*>& forest,const arma::mat& X,const Hypers& hypers) {

  arma::vec out = arma::zeros<arma::vec>(X.n_rows);
  int num_tree = forest.size();

  for(int t = 0; t < num_tree; t++) {
    out = out + predict(forest[t], X, hypers);
  }

  return out;
}

Node* birth_node(Node* tree, double* leaf_node_probability) {
  std::vector<Node*> leafs = leaves(tree);
  Node* leaf = rand(leafs);
  *leaf_node_probability = 1.0 / ((double)leafs.size());
  return leaf;
}


double LogLT(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank) 
{

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  double out;

  // Get sufficient statistics

  arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
  arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, X, hypers, mu_hat, Omega_inv,myrank);
  //cout<<"mu_hat"<<mu_hat<<"\n"<<std::endl;
  //cout<<"Omega_inv"<<Omega_inv<<"\n"<<std::endl;
  if (myrank==0)
  { 
//    int N =hypers.totalcount;
//    out=-0.5*N*log(M_2_PI*pow(hypers.sigma,2))*hypers.temperature;
    out = -0.5 * num_leaves * log(M_2_PI * pow(hypers.sigma_mu,2));
    double val, sign;
    log_det(val, sign, Omega_inv / M_2_PI);
    out -= 0.5 * val;
    out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);
  }
  else
  {
    out=0;
  }


  // Rcout << "Compute ";


  // Rcout << "Done";
  return out;

}

double LogLTA(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank,int & Mucount,double * Mulist ) 
{

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  double out;
  Mucount=num_leaves;
  

  // Get sufficient statistics

  arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
  arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, X, hypers, mu_hat, Omega_inv,myrank);
  //cout<<"mu_hat"<<mu_hat<<"\n"<<std::endl;
  //cout<<"Omega_inv"<<Omega_inv<<"\n"<<std::endl;
  if (myrank==0)
  { 
//    int N =hypers.totalcount;
//    out=-0.5*N*log(M_2_PI*pow(hypers.sigma,2))*hypers.temperature;

    arma::vec mu_samp = rmvnorm(mu_hat, Omega_inv);

    
    for(int i=0; i<Mucount; i++) 
    {
      Mulist[i]=mu_samp[i];
    }  



   
    out = -0.5 * num_leaves * log(M_2_PI * pow(hypers.sigma_mu,2));
    double val, sign;
    log_det(val, sign, Omega_inv / M_2_PI);
    out -= 0.5 * val;
    out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);


  }
  else
  {
    out=0;
  }


  // Rcout << "Compute ";


  // Rcout << "Done";
  return out;

}

double LogLTB(Node* n, const arma::vec& Y,const arma::mat& X, const Hypers& hypers,const double myrank,int & Mucount,double * Mulist ) 
{

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  double out;
  Mucount=num_leaves;
  

  // Get sufficient statistics

  arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
  arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
  GetSuffStatsB(n, Y, X, hypers, mu_hat, Omega_inv);
  //cout<<"mu_hat"<<mu_hat<<"\n"<<std::endl;
  //cout<<"Omega_inv"<<Omega_inv<<"\n"<<std::endl;

//    int N =hypers.totalcount;
//    out=-0.5*N*log(M_2_PI*pow(hypers.sigma,2))*hypers.temperature;

    arma::vec mu_samp = rmvnorm(mu_hat, Omega_inv);

    
    for(int i=0; i<Mucount; i++) 
    {
      Mulist[i]=mu_samp[i];
    }  



   
    out = -0.5 * num_leaves * log(M_2_PI * pow(hypers.sigma_mu,2));
    double val, sign;
    log_det(val, sign, Omega_inv / M_2_PI);
    out -= 0.5 * val;
    out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);





  // Rcout << "Compute ";


  // Rcout << "Done";
  return out;

}

double probability_node_birth(Node* tree) {
  return tree->is_leaf ? 1.0 : 0.5;
}


std::vector<Node*> not_grand_branches(Node* tree) {
  std::vector<Node*> ngb(0);
  not_grand_branches(ngb, tree);
  return ngb;
}


void not_grand_branches(std::vector<Node*>& ngb, Node* node) {
  if(!node->is_leaf) {
    bool left_is_leaf = node->left->is_leaf;
    bool right_is_leaf = node->right->is_leaf;
    if(left_is_leaf && right_is_leaf) {
      ngb.push_back(node);
    }
    else {
      not_grand_branches(ngb, node->left);
      not_grand_branches(ngb, node->right);
    }
  }
}

void leaves(Node* x, std::vector<Node*>& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}


std::vector<Node*> leaves(Node* x) {
  std::vector<Node*> leafs(0);
  leaves(x, leafs);
  return leafs;
}

void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,const Hypers& hypers,const double myrank) {
  unsigned int T_nid;
  int T_var;
  double T_val;

  double leaf_probability;
  Node* leaf;
  int leaf_depth;
  double leaf_prior;
  double p_forward;


  int tag=1;
  char buffer[48];
	int position=0;
  MPI_Request *request;

  int AType=0;



  if (myrank==0)
  {
    leaf_probability = 0.0;
    leaf = birth_node(tree, &leaf_probability);
    leaf_depth = depth(leaf);
    leaf_prior = growth_prior(leaf_depth, hypers);
    p_forward = log(probability_node_birth(tree) * leaf_probability);
  }
  
  double ll_before = LogLT(tree, Y, X, hypers,myrank);

  if (myrank==0)
  {
    ll_before += log(1.0 - leaf_prior);
    leaf->BirthLeaves(hypers);
    T_var=leaf->var;
    T_val=leaf->val;
    T_nid=leaf->nid;    

    request = new MPI_Request[hypers.np];
    MPI_Pack(&T_nid,1,MPI_UNSIGNED,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&T_var,1,MPI_INT,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&T_val,1,MPI_DOUBLE,buffer,48,&position,MPI_COMM_WORLD);
    for(int i=1; i<=hypers.np; i++) 
    {
      MPI_Isend(buffer,48,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
    }
    MPI_Waitall(hypers.np,request,MPI_STATUSES_IGNORE);
    delete[] request;
  }
  else
  {
    MPI_Recv(buffer,48,MPI_PACKED,0,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    position=0;
    MPI_Unpack(buffer,48,&position,&T_nid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&T_var,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&T_val,1,MPI_DOUBLE,MPI_COMM_WORLD);

    leaf=tree->getptr(T_nid);
    if (leaf->is_leaf)
    {
      leaf->AddLeaves();
      leaf->var=T_var;
      leaf->val=T_val;
      leaf->GetLimits();
    }
  }  

  double ll_after = LogLT(tree, Y, X, hypers,myrank);
  if (myrank==0)
  {
    ll_after += log(leaf_prior) +log(1.0 - growth_prior(leaf_depth + 1, hypers)) +log(1.0 - growth_prior(leaf_depth + 1, hypers));
    std::vector<Node*> ngb = not_grand_branches(tree);
    double p_not_grand = 1.0 / ((double)(ngb.size()));
    double p_backward = log((1.0 - probability_node_birth(tree)) * p_not_grand);
    double log_trans_prob = ll_after + p_backward - ll_before - p_forward;
    if(log(arma::randu()) > log_trans_prob) 
    { 
      AType=1;
      leaf->DeleteLeaves();
      leaf->var = 0;
    }
    else
    {
      AType=0;
    }  
  }
  
  MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
  if ((myrank!=0) && AType)
  {
    leaf->DeleteLeaves();
    leaf->var=0;
  }


	
}

Node* death_node(Node* tree, double* p_not_grand) {
  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);
  *p_not_grand = 1.0 / ((double)ngb.size());

  return branch;
}


void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,const Hypers& hypers,const double myrank) 
{
  unsigned int T_nid;
  int AType;

  double p_not_grand;
  Node* branch;
  int leaf_depth;
  double leaf_prob;
  double left_prior;
  double right_prior;
  double ll_before;
  double p_forward;
  Node* left;
  Node* right;
  AType=0;


  T_nid=0;
  if (myrank==0)
  {
    p_not_grand = 0.0;
    branch = death_node(tree, &p_not_grand);
    T_nid=branch->nid;  
    leaf_depth = depth(branch->left);
    leaf_prob = growth_prior(leaf_depth - 1, hypers);
    left_prior = growth_prior(leaf_depth, hypers);
    right_prior = growth_prior(leaf_depth, hypers);
  }
  ll_before = LogLT(tree, Y, X, hypers,myrank);
  if  (myrank==0)
  {
    ll_before += log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);
    p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));
    left = branch->left;
    right = branch->right;
    branch->left = branch;
    branch->right = branch;
    branch->is_leaf = true;
  }
  MPI_Bcast(&T_nid,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
  if (myrank!=0)
  {
    branch=tree->getptr(T_nid);
    left = branch->left;
    right = branch->right;
    branch->left = branch;
    branch->right = branch;
    branch->is_leaf = true;
  }
  double ll_after = LogLT(tree, Y, X, hypers,myrank) ;
  if (myrank==0)
  {
    ll_after = ll_after +log(1.0 - leaf_prob);
    std::vector<Node*> leafs = leaves(tree);
    double p_backwards = log(1.0 / ((double)(leafs.size())) * probability_node_birth(tree));
    double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;

    if(log(arma::randu()) > log_trans_prob) 
    {
      branch->left = left;
      branch->right = right;
      branch->is_leaf = false;
      AType=1;
    }
    else 
    {
      delete left;
      delete right;
      AType=0;
    }
  }

  MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
  if (myrank!=0) 
  {
    if(AType)
    {
        branch->left = left;
        branch->right = right;
        branch->is_leaf = false;
    }
    else
    {
        delete left;
        delete right;
    }
  }
}

void change_decision_rule(Node* tree, const arma::mat& X, const arma::vec& Y, const Hypers& hypers,const double myrank) 
{
  Node* branch;
  unsigned int T_nid;
  int N_var;
  double N_val;
  double N_lower;
  double N_upper;
  MPI_Request *request;


  int    O_var;
  double O_val;
  double O_lower;
  double O_upper;


  int tag=3;
  char buffer[48];
  int position=0;
  
  int AType=0;

  if(myrank==0)
  {
    std::vector<Node*> ngb = not_grand_branches(tree);
    branch = rand(ngb);
    T_nid=branch->nid;
  }
  double ll_before = LogLT(tree, Y, X, hypers,myrank);
  if (myrank==0)
  {
    O_var=branch->var;
    O_val=branch->val;
    O_lower = branch->lower;
    O_upper = branch->upper;
    branch->var = hypers.SampleVar();
    branch->GetLimits();
    branch->val = (branch->upper - branch->lower) * arma::randu() + branch->lower;

    N_var  =branch->var;
    N_val  =branch->val;
    N_lower=branch->lower;
    N_upper=branch->upper;

    request = new MPI_Request[hypers.np];
    MPI_Pack(&T_nid,1,MPI_UNSIGNED,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&N_var,1,MPI_INT,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&N_val,1,MPI_DOUBLE,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&N_lower,1,MPI_DOUBLE,buffer,48,&position,MPI_COMM_WORLD);
    MPI_Pack(&N_upper,1,MPI_DOUBLE,buffer,48,&position,MPI_COMM_WORLD);
    

    for(int i=1; i<=hypers.np; i++) 
    {
    MPI_Isend(buffer,48,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
    }
    MPI_Waitall(hypers.np,request,MPI_STATUSES_IGNORE);
    delete[] request;

  }
  else
  {
    MPI_Recv(buffer,48,MPI_PACKED,0,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    position=0;
    MPI_Unpack(buffer,48,&position,&T_nid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&N_var,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&N_val,1,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&N_lower,1,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,48,&position,&N_upper,1,MPI_DOUBLE,MPI_COMM_WORLD);
    
    branch=tree->getptr(T_nid);
    O_var=branch->var;
    O_val=branch->val;
    O_lower = branch->lower;
    O_upper = branch->upper;

    branch->var=N_var;
    branch->val=N_val;
    branch->lower=N_lower;
    branch->upper=N_upper;
  }
  double ll_after = LogLT(tree, Y, X, hypers,myrank);
  if (myrank==0)
   {
        double log_trans_prob = ll_after - ll_before;
        
        if(log(arma::randu()) > log_trans_prob) 
        {
            branch->var = O_var;
            branch->val = O_val;
            branch->lower = O_lower;
            branch->upper = O_upper;
            AType=1;
        }
        else
        {
            AType=0;
        }

   }
   MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
   if ((myrank!=0) && AType)
   {
        branch->var = O_var;
        branch->val = O_val;
        branch->lower = O_lower;
        branch->upper = O_upper;
   }


}

void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank) {

  double MH_BD = 0.7;
  int AType=0;
  double rAct;

  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) {
    arma::vec Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::vec res = Y - Y_star;

    if (myrank==0)
    { 
      rAct=arma::randu();
      if (forest[t]->is_leaf)
      {
        AType=1;
      }
      else if (rAct<MH_BD/2)
      {
        AType=1;
      }
      else if (rAct<MH_BD)
      {
        AType=2;
      }
      else
      {
        AType=3;
      }
    }
    
    MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);

    if (AType==1)
    {
      node_birth(forest[t], X, res, hypers,myrank);
    }
    else if (AType==2)
    {
      node_death(forest[t], X, res, hypers,myrank);
    }
    else if (AType==3)
    {
      change_decision_rule(forest[t], X, res, hypers,myrank);
    }
    else
    {
      cout<< "\n"<<myrank <<"Wrong Type " << AType <<std::ends;
    }




    if(opts.update_tau) forest[t]->UpdateTau(res, X, hypers,myrank);


    forest[t]->UpdateMu(res, X, hypers,myrank);

    Y_hat = Y_star + predict(forest[t], X, hypers);
  }
}







void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat, Hypers& hypers, const arma::mat& X, const arma::vec& Y, const Opts& opts,const double myrank) {
  
  TreeBackfitOne(forest, Y_hat, hypers, X, Y, opts,myrank);
  if (!(hypers.binary))
  {
  arma::vec res = Y - Y_hat;
  hypers.UpdateSigma(res,myrank);
  }


}

arma::vec get_means(std::vector<Node*>& forest) {
  std::vector<double> means(0);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_means(forest[t], means);
  }
  arma::vec out(&(means[0]), means.size());
  return out;
}


void get_means(Node* node, std::vector<double>& means) {

  if(node->is_leaf) {
    means.push_back(node->mu);
  }
  else {
    get_means(node->left, means);
    get_means(node->right, means);
  }
}


Node* rand(std::vector<Node*> ngb) {

  int N = ngb.size();
  arma::vec p = arma::ones<arma::vec>(N) / ((double)(N));
  int i = sample_class(p);
  return ngb[i];
}


arma::vec loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers) {
  arma::vec res = Y - Y_hat;
  arma::vec out = arma::zeros<arma::vec>(Y.size());
  for(unsigned int i = 0; i < Y.size(); i++) {
    out(i) = -0.5 * log(M_2_PI * pow(hypers.sigma,2)) - 0.5 * pow(res(i) / hypers.sigma, 2);
  }
  return out;
}

int main(int argc, char** argv)
{ 
  std::stringstream srnstr; 
  std::stringstream tempfnss;  
  Opts TOpts;
  Hypers THypers;
  arma::vec Y;
  arma::vec MM;
  arma::mat X;  
  arma::mat X_test;   
  arma::vec Y_hat;
  arma::vec RY;
  bool verbose;
  int myrank, processes;
  std::vector<Node*> forest; 
  double n_local;
  double n_global;



  double T_PrP[2] = {0.0,0.0};






  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&processes);




  srnstr<< "\nMPI: node " << myrank << " of " << processes << " processes.\n"<<std::ends;
  cout << srnstr.str()<<std::flush;

  if(argc!=29) {cout << "Parameter not match\n"; return 1;}


  arma::arma_rng::set_seed(myrank+888);  

	// srnstr.str("");
	// srnstr << "\n"<<myrank <<"RNG seed set to " << myrank <<std::ends;
	// cout << srnstr.str()<<std::flush;





  // srnstr.str("");
  // for(int i=0;i<argc;i++)
  // {
  //   srnstr <<i<<" "<<argv[i]<<"\n";
  // }
  // cout << srnstr.str()<<std::flush;

	// std::string tempfns; 
	// tempfnss.str("");
	// tempfnss << "para" <<myrank<< ".txt";
  // tempfns = tempfnss.str();
  // std::ofstream tf(tempfns.c_str());





  THypers.alpha=atof(argv[2]);
  THypers.beta=atof(argv[3]);
  THypers.gamma=atof(argv[4]);
  THypers.np=processes-1;
  THypers.binary=atoi(argv[5]);
  THypers.sigma_hat=atof(argv[6]);
  THypers.shape=atof(argv[7]);
  THypers.width=atof(argv[8]);
  THypers.num_tree=ceil(atoi(argv[9])/processes)*processes;
  THypers.alpha_scale=atof(argv[10]);
  THypers.alpha_shape_1=atof(argv[11]);
  THypers.alpha_shape_2=atof(argv[12]);
  THypers.tau_rate=atof(argv[13]);
  THypers.binaryOffset=atof(argv[14]);
  THypers.temperature=atof(argv[15]);
  THypers.sigma_mu=atof(argv[16]);

  tempfnss.str("");
  tempfnss << argv[1] << "group.txt";
  THypers.group.load(tempfnss.str(),arma::raw_ascii);
  InitHypers(THypers);

  TOpts.num_burn=atoi(argv[17]);
  TOpts.num_thin=atoi(argv[18]);
  TOpts.num_save=atoi(argv[19]);
  TOpts.num_print=atoi(argv[20]);
  TOpts.update_sigma_mu=atoi(argv[21]);
  TOpts.update_s=atoi(argv[22]);
  TOpts.update_alpha=atoi(argv[23]);
  TOpts.update_beta=atoi(argv[24]);
  TOpts.update_gamma=atoi(argv[25]);
  TOpts.update_tau=atoi(argv[26]);
  TOpts.update_tau_mean=atoi(argv[27]);
  verbose=atoi(argv[28]);


  if (myrank==0)
  {
    srnstr.str("");
    srnstr <<"num_burn        "<<TOpts.num_burn<<"\n";
    srnstr <<"num_thin        "<<TOpts.num_thin<<"\n";
    srnstr <<"num_save        "<<TOpts.num_save<<"\n";
    srnstr <<"num_print       "<<TOpts.num_print<<"\n";		
    srnstr <<"update_sigma_mu "<<TOpts.update_sigma_mu<<"\n";
    srnstr <<"update_s        "<<TOpts.update_s<<"\n";
    srnstr <<"update_alpha    "<<TOpts.update_alpha<<"\n";
    srnstr <<"update_beta     "<<TOpts.update_beta<<"\n";
    srnstr <<"update_gamma    "<<TOpts.update_gamma<<"\n";
    srnstr <<"update_tau      "<<TOpts.update_tau<<"\n";		
    srnstr <<"update_tau_mean "<<TOpts.update_tau_mean<<"\n";	
    srnstr <<"verbose               "<<verbose<<"\n";	 
    srnstr <<THypers.s<<"\n";  
    srnstr <<THypers.logs<<"\n";      
    srnstr <<"alpha "<< 	THypers.alpha<<"\n"
    <<"beta "<< 	THypers.beta<<"\n"
    <<"gamma "<< 	THypers.gamma<<"\n"
    <<"sigma_hat "<< 	THypers.sigma_hat<<"\n"
    <<"shape "<< 	THypers.shape<<"\n"
    <<"width "<< 	THypers.width<<"\n"
    <<"num_tree "<< 	THypers.num_tree<<"\n"
    <<"alpha_scale "<< 	THypers.alpha_scale<<"\n"
    <<"alpha_shape_1 "<< 	THypers.alpha_shape_1<<"\n"
    <<"alpha_shape_2 "<< 	THypers.alpha_shape_2 <<"\n"         
    <<"tau_rate "<< 	THypers.tau_rate<<"\n"
    <<"binaryOffset "<< 	THypers.binaryOffset<<"\n"
    <<"temperature "<< 	THypers.temperature<<"\n"
    <<"num_groups "<< 	THypers.num_groups<<"\n"
    <<"sigma_mu_hat "<< 	THypers.sigma_mu_hat<<"\n"
    <<"sigma_mu "<< 	THypers.sigma_mu<<"\n";
    cout << srnstr.str()<<std::flush;
  }



    
  tempfnss.str("");
  tempfnss << argv[1] << "MM"<<".csv";
  MM.load(tempfnss.str(),arma::csv_ascii);
  //cout<<"MM"<<MM.n_elem<<"\n";

  tempfnss.str("");
  tempfnss << argv[1] << "x"<<myrank<<".csv";
  X.load(tempfnss.str(),arma::csv_ascii);
  //cout<<"x"<<X.n_rows<<" "<<X.n_cols<<"\n";

  tempfnss.str("");
  tempfnss << argv[1] << "xp"<<myrank<<".csv";
  X_test.load(tempfnss.str(),arma::csv_ascii);
  //cout<<"x_test"<<X_test.n_rows<<" "<<X_test.n_cols<<"\n";

  forest = init_forest(X, Y, THypers);
  Y_hat = arma::zeros<arma::vec>(X.n_rows);
  //cout<<"Y_hat"<<Y_hat.n_elem<<"\n";

  n_local=X.n_rows;

  tempfnss.str("");
  tempfnss << argv[1] << "y"<<myrank<<".csv";
  RY.load(tempfnss.str(),arma::csv_ascii);

  Y = arma::zeros<arma::vec>(X.n_rows);
  if (!(THypers.binary))
  {
    Y=RY*1.0;
  }
  else
  {
    for(size_t k=0; k<n_local; k++) 
    {
      if(RY[k]==0) Y[k]= -1*rtnorm(0., THypers.binaryOffset, 1.0);
      else Y[k]=rtnorm(0., -1*THypers.binaryOffset, 1.0);
    }
  }

  n_global=0; 
  MPI_Reduce(&n_local, &n_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myrank==0)
  {
    THypers.totalcount=n_global;
  }

  arma::mat sigma = arma::zeros<arma::mat>(TOpts.num_burn+TOpts.num_save,2);
  arma::mat para=arma::zeros<arma::mat>(TOpts.num_save,4);
  arma::mat Y_hat_train = arma::zeros<arma::mat>(TOpts.num_save, X.n_rows);
  arma::mat Y_hat_test = arma::zeros<arma::mat>(TOpts.num_save, X_test.n_rows);
  arma::umat var_counts = arma::zeros<arma::umat>(TOpts.num_save, THypers.s.size());  
  arma::mat Y_train_ave = arma::zeros<arma::mat>(X.n_rows, 1);  
  arma::mat Y_test_ave = arma::zeros<arma::mat>(X_test.n_rows, 1);  
  
  clock_t start = clock();



  for(int i = 0; i < TOpts.num_burn; i++) 
  { 
    IterateGibbsNoS(forest, Y_hat, THypers, X, Y, TOpts,myrank);
    
    if (myrank==0) 
    { 
      arma::vec means = get_means(forest);  
      if(TOpts.update_sigma_mu) 
      {
          THypers.UpdateSigmaMu(means);
      }

      if(TOpts.update_beta) THypers.UpdateBeta(forest);
      if(TOpts.update_gamma) THypers.UpdateGamma(forest);
      if(TOpts.update_tau_mean) THypers.UpdateTauRate(forest);
      sigma(i,0) = THypers.sigma;
      sigma(i,1) =  THypers.sigma_mu; 
      if(i >= TOpts.num_burn / 2) 
      {
        if(TOpts.update_s) UpdateS(forest, THypers);
        if(TOpts.update_alpha) THypers.UpdateAlpha();     
      }

      T_PrP[0]=THypers.sigma_mu;
      T_PrP[1]=THypers.alpha;


    }

    MPI_Bcast(T_PrP,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (myrank!=0)
    {
      THypers.sigma_mu=T_PrP[0];
      THypers.alpha=T_PrP[1];
    }



    // tf <<"alpha ,"<< 	THypers.alpha<<","
    //      <<"beta ,"<< 	THypers.beta<<","
    //      <<"gamma ,"<< 	THypers.gamma<<","
    //      <<"sigma_hat ,"<< 	THypers.sigma_hat<<","
    //      <<"shape ,"<< 	THypers.shape<<","
    //      <<"width ,"<< 	THypers.width<<","
    //      <<"num_tree ,"<< 	THypers.num_tree<<","
    //      <<"alpha_scale ,"<< 	THypers.alpha_scale<<","
    //      <<"alpha_shape_1 ,"<< 	THypers.alpha_shape_1<<","
    //      <<"alpha_shape_2 ,"<< 	THypers.alpha_shape_2 <<","         
    //      <<"tau_rate ,"<< 	THypers.tau_rate<<","
    //      <<"num_tree_prob ,"<< 	THypers.num_tree_prob<<","
    //      <<"temperature ,"<< 	THypers.temperature<<","
    //      <<"num_groups ,"<< 	THypers.num_groups<<","
    //      <<"sigma_mu_hat ,"<< 	THypers.sigma_mu_hat<<","
    //      <<"sigma_mu ,"<< 	THypers.sigma_mu<< endl;

    if (THypers.binary)
    {
      for(size_t k=0; k<n_local; k++) 
      {
      if(RY[k]==0) Y[k]= -1.0*rtnorm(-1*Y_hat[k], THypers.binaryOffset, 1.0);
      else Y[k]=rtnorm(Y_hat[k], -1*THypers.binaryOffset, 1.0);
      }
    }




    if( ((i + 1) % TOpts.num_print == 0))
    {
      srnstr.str("");
      srnstr <<"Machine"<<myrank<< " Finishing Burning " << i + 1 << "\n";
      cout << srnstr.str()<<std::flush;
    }      
  }

  //  arma::mat s = arma::zeros<arma::mat>(TOpts.num_save, THypers.s.size());
  //  arma::mat loglik_train = arma::zeros<arma::mat>(TOpts.num_save, Y_hat.size());


  for(int i = 0; i < TOpts.num_save; i++) 
  {
    for(int b = 0; b < TOpts.num_thin; b++) 
    {
      IterateGibbsNoS(forest, Y_hat, THypers, X, Y, TOpts,myrank);
      if (myrank==0)
      {
        arma::vec means = get_means(forest);  
        if(TOpts.update_sigma_mu) 
        {
            THypers.UpdateSigmaMu(means);
        }

          
        if(TOpts.update_beta) THypers.UpdateBeta(forest);
        if(TOpts.update_gamma) THypers.UpdateGamma(forest);
        if(TOpts.update_tau_mean) THypers.UpdateTauRate(forest);        
        if(TOpts.update_s) UpdateS(forest, THypers);
        if(TOpts.update_alpha) THypers.UpdateAlpha();  


        T_PrP[0]=THypers.sigma_mu;
        T_PrP[1]=THypers.alpha;          
        
      }




      MPI_Bcast(T_PrP,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      if (myrank!=0)
      {
        THypers.sigma_mu=T_PrP[0];
        THypers.alpha=T_PrP[1];
      }

      if (THypers.binary)
      {
        for(size_t k=0; k<n_local; k++) 
        {
        if(RY[k]==0) Y[k]= -1.0*rtnorm(-1*Y_hat[k], THypers.binaryOffset, 1.0);
        else Y[k]=rtnorm(Y_hat[k], -1*THypers.binaryOffset, 1.0);
        }
      }
    }







    // tf <<"alpha ,"<< 	THypers.alpha<<","
    //      <<"beta ,"<< 	THypers.beta<<","
    //      <<"gamma ,"<< 	THypers.gamma<<","
    //      <<"sigma_hat ,"<< 	THypers.sigma_hat<<","
    //      <<"shape ,"<< 	THypers.shape<<","
    //      <<"width ,"<< 	THypers.width<<","
    //      <<"num_tree ,"<< 	THypers.num_tree<<","
    //      <<"alpha_scale ,"<< 	THypers.alpha_scale<<","
    //      <<"alpha_shape_1 ,"<< 	THypers.alpha_shape_1<<","
    //      <<"alpha_shape_2 ,"<< 	THypers.alpha_shape_2 <<","         
    //      <<"tau_rate ,"<< 	THypers.tau_rate<<","
    //      <<"num_tree_prob ,"<< 	THypers.num_tree_prob<<","
    //      <<"temperature ,"<< 	THypers.temperature<<","
    //      <<"num_groups ,"<< 	THypers.num_groups<<","
    //      <<"sigma_mu_hat ,"<< 	THypers.sigma_mu_hat<<","
    //      <<"sigma_mu ,"<< 	THypers.sigma_mu<< endl;



    
    if (myrank==0)
    {
    sigma(i+ TOpts.num_burn,0) = THypers.sigma;
    sigma(i+ TOpts.num_burn,1) =  THypers.sigma_mu; 
    var_counts.row(i) = arma::trans(get_var_counts(forest, THypers));
    para(i,0)  = THypers.alpha;
    para(i,1)  = THypers.beta;
    para(i,2)  = THypers.gamma;
    para(i,3)  = THypers.tau_rate;

    }

    if (THypers.binary)
    {
    Y_hat_train.row(i) = arma::trans(arma::normcdf(Y_hat+THypers.binaryOffset));
    Y_hat_test.row(i) =  arma::trans(arma::normcdf(predict(forest, X_test, THypers)+THypers.binaryOffset)) ;       
    }
    else
    {
    Y_hat_train.row(i) = Y_hat.t();
    Y_hat_test.row(i) =  arma::trans(predict(forest, X_test, THypers))  ;
    }






    if( ((i + 1) % TOpts.num_print == 0))
    {
      srnstr.str("");
      srnstr <<"Machine"<<myrank<< " Finishing save " << i + 1 << "\n";
      cout << srnstr.str()<<std::flush;
    }
  }
  clock_t finish = clock();
  double consumeTime = (double)(finish-start)/CLOCKS_PER_SEC;

  srnstr.str("");
  srnstr <<"Machine"<<myrank<< " running " << consumeTime << "Seconds\n";
  cout << srnstr.str()<<std::flush;

  if (THypers.binary)
  {
  Y_train_ave.col(0)=arma::trans(arma::mean(Y_hat_train,0)) ;
  }
  else
  {
  Y_train_ave.col(0)=(MM(0)-MM(1))*(arma::trans(arma::mean(Y_hat_train,0))+0.5)+MM(1);
  }

  tempfnss.str("");
  tempfnss << argv[1] << "_R_ytrain"<<myrank<<".csv";
  Y_train_ave.save(tempfnss.str(),arma::csv_ascii);


  if (THypers.binary)
  {
  Y_test_ave.col(0)=arma::trans(arma::mean(Y_hat_test,0)) ;
  }
  else
  {
  Y_test_ave.col(0)=(MM(0)-MM(1))*(arma::trans(arma::mean(Y_hat_test,0))+0.5)+MM(1);
  }
  tempfnss.str("");
  tempfnss << argv[1] << "_R_ytest"<<myrank<<".csv";

  Y_test_ave.save(tempfnss.str(),arma::csv_ascii);
  if (myrank==0)
  {
  tempfnss.str("");
  tempfnss << argv[1] << "_R_vcount"<<".csv";
  var_counts.save(tempfnss.str(),arma::csv_ascii);

  tempfnss.str("");
  tempfnss << argv[1] << "_R_para"<<".csv";
  para.save(tempfnss.str(),arma::csv_ascii);    

  sigma=sigma*(MM(0)-MM(1));
  tempfnss.str("");
  tempfnss << argv[1] << "_R_sigma"<<".csv";
  sigma.save(tempfnss.str(),arma::csv_ascii);  
  }







	






  //   for(unsigned int b = 0; b < forest.size(); b++) 
  //   {
  //     std::vector<Node*> leafs = leaves(forest[b]);
  //     for (unsigned int a=0;a<leafs.size();a++)
  //     {
  //       tf <<b<<","<<a<<"," << leafs[a]->nid <<","<<leafs[a]->mu<<","<<leafs[a]->tau<< endl;
  //     }
  //   }

 




  if (verbose)
  {
    // srnstr.str("");
    // srnstr <<"X"<<X.n_rows<<" "<<X.n_cols  << "  "  << lbeta(1.5,2.3)<<"\n";  
    // srnstr <<"Y"<<Y.n_elem<< "\n";   											
    // cout << srnstr.str()<<std::flush;  
  }

  MPI_Finalize();
  return 0;

}	














void TreeBackfitOne(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank) 
{

  double MH_BD = 0.7;
  int AType=0;
  int DType=0;
  double rAct;

  int leafsize;
  int leafsizeA;



  double l_prob;
  Node* OneNode;
  unsigned int OneID;
  double Node_depth;
  double T_tau;
  int T_var;
  double T_val;
  double NgbCount;
  int loc;


  double leaf_prior;
  double ll_before=0;
  double ll_after=0;
  double LogLT1=0;
  double LogLT2=0;
  double p_backwards=0;
  double p_forward=0;

  double leaf_prob;

 







  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) 
  { 

    arma::vec Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::vec res = Y - Y_star;
    std::vector<Node*> leafs = leaves(forest[t]);  
    leafsize=(int)(leafs.size());   

    arma::vec mu_samp;
    double LLogLT=0;

    std::vector<unsigned int> leafsID(leafsize);
    double * MUlist=new double [leafsize+1];

    for (int i=0;i<leafsize;i++)
    { 
      leafsID[i]=leafs[i]->nid;
    }

    if (myrank==0)
    { 
      rAct=arma::randu();
      if (forest[t]->is_leaf)
      {
        AType=1;
      }
      else if (rAct<MH_BD/2)
      {
        AType=1;
      }
      else if (rAct<MH_BD)
      {
        AType=2;
      }
      else
      {
        AType=3;
      }


      if (AType==1)
      {

        l_prob = 0.0;
        OneNode = birth_node(forest[t], &l_prob);
        OneID=OneNode->nid;
        Node_depth = depth(OneNode);
        leaf_prior = growth_prior(Node_depth, hypers);
        p_forward = log(probability_node_birth(forest[t]) * l_prob);

        ll_before = log(1.0 - leaf_prior);
        T_tau=RandExp(hypers.width);
        T_var = hypers.SampleVar();
        T_val=  OneNode->Getval(T_var);
        ll_after = log(leaf_prior) +log(1.0 - growth_prior(Node_depth + 1, hypers)) +log(1.0 - growth_prior(Node_depth + 1, hypers));
        NgbCount=1.0;
        for (int i=0;i<leafsize-1;i++)
        { 
          if (((leafsID[i] +1) % 2) &&   (!(leafsID[i+1]-leafsID[i]-1)) && (leafsID[i]!=OneID) &&  (leafsID[i+1]!=OneID)  )
          {
            NgbCount++;
          }
        }
        p_backwards = log(0.5 / NgbCount);
        
        for (loc=0;loc<leafsize;loc++)
        { 
          if (leafsID[loc]==OneID)  break;
        }


      }
      else if (AType==2)
      {
        l_prob = 0.0;
        OneNode = death_node(forest[t], &l_prob);
        OneID=OneNode->nid;  
        Node_depth = depth(OneNode);
        leaf_prob = growth_prior(Node_depth, hypers);
        ll_before = 2*log(1.0 - growth_prior(Node_depth+1, hypers)) + log(leaf_prob);

        ll_after = log(1.0 - leaf_prob);
        p_forward = log(l_prob * 0.5 );
        p_backwards = log(1.0 / (leafsize-1) * ((leafsize==2)? 1.0 : 0.5));
        T_var=0;
        T_val=0;
        T_tau=0;


        for (loc=0;loc<leafsize;loc++)
        { 
          if (leafsID[loc]==2*OneID)  break;
        }
      }
      else if (AType==3)
      {
        std::vector<Node*> ngb = not_grand_branches(forest[t]);
        OneNode = rand(ngb);
        OneID=OneNode->nid;  
        ll_before = 0;
        ll_after = 0;
        T_var = hypers.SampleVar();
        T_val=  OneNode->Getval(T_var);
        T_tau=RandExp(hypers.width);
        p_forward=0;
        p_backwards = 0;
        for (loc=0;loc<leafsize;loc++)
        { 
          if (leafsID[loc]==2*OneID)  break;
        }


      }
    }






    MPI_Bcast(&AType,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&OneID,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    MPI_Bcast(&loc,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&T_var,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&T_val,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&T_tau,1,MPI_DOUBLE,0,MPI_COMM_WORLD);     

    leafsizeA=0;
    if (AType==1)
    {
      leafsizeA = leafsize+1;
    } 
    else if  (AType==2)
    {
      leafsizeA = leafsize-1;
    }    
    else if  (AType==3)
    {
      leafsizeA = leafsize;
    }  


    






    arma::vec w_i_1 = arma::zeros<arma::vec>(leafsize);
    arma::vec mu_hat_S_1 = arma::zeros<arma::vec>(leafsize);
    arma::mat Lambda_S_1 = arma::zeros<arma::mat>(leafsize, leafsize);
    arma::vec mu_hat_H_1 = arma::zeros<arma::vec>(leafsize);
    arma::mat Lambda_H_1 = arma::zeros<arma::mat>(leafsize, leafsize);

    arma::vec w_i_2 = arma::zeros<arma::vec>(leafsizeA);
    arma::vec mu_hat_S_2 = arma::zeros<arma::vec>(leafsizeA);
    arma::mat Lambda_S_2 = arma::zeros<arma::mat>(leafsizeA, leafsizeA);
    arma::vec mu_hat_H_2 = arma::zeros<arma::vec>(leafsizeA);
    arma::mat Lambda_H_2 = arma::zeros<arma::mat>(leafsizeA, leafsizeA);


    double TW;
    double LW;




    for(unsigned int i = 0; i < X.n_rows; i++) 
    {
      forest[t]->GetW(X, i);
      for(int j = 0; j < leafsize; j++) 
      {
        w_i_1(j) = leafs[j]->current_weight;
      }

      if (AType==1)
      {
        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            LW=activation(X(i,T_var), T_val, forest[t]->tau);
            //LW=activation(X(i,T_var), T_val, T_tau); 
            w_i_2(j)=w_i_1(j)*LW;
            w_i_2(j+1)=w_i_1(j)*(1-LW);
            j++;
          }
          else
          {
            w_i_2(j)=w_i_1(j-1);
          }
        }
         
      } 
      else if  (AType==2)
      {

        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            w_i_2(j)=w_i_1(j)+w_i_1(j+1);
          }
          else
          {
            w_i_2(j)=w_i_1(j+1);
          }
        }
      }    
      else if  (AType==3)
      {
        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            TW=(forest[t]->getptr(OneID))->current_weight;
            
            LW=activation(X(i,T_var), T_val, forest[t]->tau);
            //LW=activation(X(i,T_var), T_val, T_tau); 
            w_i_2(j)=LW*TW;
            w_i_2(j+1)=TW*(1-LW);  
            j++;
          }
          else
          {
            w_i_2(j)=w_i_1(j);
          }
        }        
      }  
     



      mu_hat_S_1 = mu_hat_S_1 + res(i) * w_i_1;
      Lambda_S_1 = Lambda_S_1 + w_i_1 * arma::trans(w_i_1);

      mu_hat_S_2 = mu_hat_S_2 + res(i) * w_i_2;
      Lambda_S_2 = Lambda_S_2 + w_i_2 * arma::trans(w_i_2);

    }

    double* mu_hat_S_mem_1 = mu_hat_S_1.memptr();
    double* Lambda_S_mem_1 = Lambda_S_1.memptr();
    double* mu_hat_H_mem_1 = mu_hat_H_1.memptr();
    double* Lambda_H_mem_1 = Lambda_H_1.memptr();


    double* mu_hat_S_mem_2 = mu_hat_S_2.memptr();
    double* Lambda_S_mem_2 = Lambda_S_2.memptr();
    double* mu_hat_H_mem_2 = mu_hat_H_2.memptr();
    double* Lambda_H_mem_2 = Lambda_H_2.memptr();


    MPI_Reduce(mu_hat_S_mem_1, mu_hat_H_mem_1, leafsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Lambda_S_mem_1, Lambda_H_mem_1, leafsize*leafsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(mu_hat_S_mem_2, mu_hat_H_mem_2, leafsizeA, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Lambda_S_mem_2, Lambda_H_mem_2, leafsizeA*leafsizeA, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   



 
 
 
    if (myrank==0)
    {
      Lambda_H_1 = Lambda_H_1 / pow(hypers.sigma, 2) * hypers.temperature;
      mu_hat_H_1 = mu_hat_H_1 / pow(hypers.sigma, 2) * hypers.temperature;
      
      arma::mat Omega_inv1 = Lambda_H_1 + arma::eye(leafsize, leafsize) / pow(hypers.sigma_mu, 2);
      arma::vec mu_hat1 = solve(Omega_inv1, mu_hat_H_1);


      Lambda_H_2 = Lambda_H_2 / pow(hypers.sigma, 2) * hypers.temperature;
      mu_hat_H_2 = mu_hat_H_2 / pow(hypers.sigma, 2) * hypers.temperature;

      arma::mat Omega_inv2 = Lambda_H_2 + arma::eye(leafsizeA, leafsizeA) / pow(hypers.sigma_mu, 2);
      arma::vec mu_hat2 = solve(Omega_inv2, mu_hat_H_2);


      double val, sign;
      LogLT1 = -0.5 * leafsize * log(M_2_PI * pow(hypers.sigma_mu,2));
      log_det(val, sign, Omega_inv1 / M_2_PI);
      LogLT1 -= 0.5 * val;
      LogLT1 += 0.5 * dot(mu_hat1, Omega_inv1 * mu_hat1);

      LogLT2 = -0.5 * leafsizeA * log(M_2_PI * pow(hypers.sigma_mu,2));
      log_det(val, sign, Omega_inv2 / M_2_PI);
      LogLT2 -= 0.5 * val;
      LogLT2 += 0.5 * dot(mu_hat2, Omega_inv2 * mu_hat2);



      double log_trans_prob = ll_after + LogLT2 + p_backwards - ll_before - p_forward-LogLT1;

      if(log(arma::randu()) > log_trans_prob) 
      { 
        DType=0;
        LLogLT=LogLT1;  
        mu_samp = rmvnorm(mu_hat1, Omega_inv1); 
        for(int j = 0; j < leafsize; j++) 
        {
          MUlist[j]=mu_samp[j];
        }

      }
      else
      {
        DType=1;
        LLogLT=LogLT2;
        mu_samp = rmvnorm(mu_hat2, Omega_inv2);      
        for(int j = 0; j < leafsizeA; j++) 
        {
          MUlist[j]=mu_samp[j];
        }

      }
    }

    MPI_Bcast(&DType,1,MPI_INT,0,MPI_COMM_WORLD);

    if (DType)
    { 
      
      if (AType==1)
      {
        OneNode=forest[t]->getptr(OneID);
        OneNode->AddLeaves();
        OneNode->var=T_var;
        OneNode->val=T_val;
        OneNode->GetLimits();
        //OneNode->SetOneTau(T_tau);
      }
      else if  (AType==2)
      {
        OneNode=forest[t]->getptr(OneID);
        OneNode->DeleteLeaves();
      }
      else if (AType==3)
      {
        OneNode=forest[t]->getptr(OneID);

        OneNode->var = T_var;
        OneNode->GetLimits();
        OneNode->val = T_val;
        //OneNode->SetOneTau(T_tau);        
      }
    }

   if(opts.update_tau) forest[t]->UpdateTauA(res, X, hypers,myrank,LLogLT,MUlist,leafsize+1);


    forest[t]->UpdateMuA(MUlist,myrank);

    Y_hat = Y_star + predict(forest[t], X, hypers);
    delete [] MUlist;
  }
}



void TreeBackfit_Par(std::vector<Node*>& forest, arma::vec& Y_hat,const Hypers& hypers, const arma::mat& X, const arma::vec& Y,const Opts& opts,const double myrank) 
{
  std::stringstream srnstr; 

  double MH_BD = 0.7;
  int AType=0;
  int DType=0;
  double rAct;

  int leafsize;
  int leafsizeA;



  double l_prob;
  Node* OneNode;
  unsigned int OneID;
  double Node_depth;
  double T_tau;
  int T_var;
  double T_val;
  double NgbCount;
  int loc;


  double leaf_prior;
  double ll_before=0;
  double ll_after=0;
  double LogLT1=0;
  double LogLT2=0;
  double p_backwards=0;
  double p_forward=0;

  double leaf_prob;
  int num_tree = hypers.num_tree;

  int TreeWave=num_tree/(hypers.np +1);
  unsigned int Tter[num_tree];
  unsigned int NowTree;


  int Pleafsize;
  int Fleafsize;
  int Mleafsize;

  if (myrank==0)
  {
    arma::uvec ptree = arma::linspace<arma::uvec>(0, num_tree-1, num_tree);
    ptree= arma::shuffle(ptree);
    for (int t1=0;t1<num_tree;t1++)
    {
      Tter[t1]=ptree(t1);
    }
  }
   
  MPI_Bcast(Tter,num_tree,MPI_UNSIGNED,0,MPI_COMM_WORLD);
  arma::vec res; 
  for(int t1 = 0; t1 < TreeWave; t1++)
  {
    arma::vec Y_Block_S=arma::zeros<arma::vec>(Y.n_elem);
    
    // srnstr.str("");
    // srnstr<<"Myrank"<<myrank<< " " <<t1<<"Start Block\n";
    // cout << srnstr.str()<<std::flush;  


    for (int t2=0;t2<=hypers.np;t2++)
    { 
      arma::vec Y_predict=predict(forest[Tter[t1*(hypers.np +1)+t2]], X, hypers);
      Y_Block_S=Y_Block_S+Y_predict;
      if(t2==myrank)
      { 
        
        res = (Y -Y_hat)/(hypers.np +1.0) + Y_predict;
      }
    }

    NowTree=Tter[(unsigned int)(t1*(hypers.np +1)+myrank)];






    std::vector<Node*> leafs = leaves(forest[NowTree]);  
    leafsize=(int)(leafs.size());   

    arma::vec mu_samp;
    double LLogLT=0;

    std::vector<unsigned int> leafsID(leafsize);
    double * MUlist=new double [leafsize+1];

    for (int i=0;i<leafsize;i++)
    { 
      leafsID[i]=leafs[i]->nid;
    }

    rAct=arma::randu();
    if (forest[NowTree]->is_leaf)
    {
      AType=1;
    }
    else if (rAct<MH_BD/2)
    {
      AType=1;
    }
    else if (rAct<MH_BD)
    {
      AType=2;
    }
    else
    {
      AType=3;
    }


    if (AType==1)
    {
      l_prob = 0.0;
      OneNode = birth_node(forest[NowTree], &l_prob);
      OneID=OneNode->nid;
      Node_depth = depth(OneNode);
      leaf_prior = growth_prior(Node_depth, hypers);
      p_forward = log(probability_node_birth(forest[NowTree]) * l_prob);

      ll_before = log(1.0 - leaf_prior);
      T_tau=RandExp(hypers.width);
      T_var = hypers.SampleVar();
      T_val=  OneNode->Getval(T_var);
      ll_after = log(leaf_prior) +log(1.0 - growth_prior(Node_depth + 1, hypers)) +log(1.0 - growth_prior(Node_depth + 1, hypers));
      NgbCount=1.0;
      for (int i=0;i<leafsize-1;i++)
      { 
        if (((leafsID[i] +1) % 2) &&   (!(leafsID[i+1]-leafsID[i]-1)) && (leafsID[i]!=OneID) &&  (leafsID[i+1]!=OneID)  )
        {
          NgbCount++;
        }
      }
      p_backwards = log(0.5 / NgbCount);
      
      for (loc=0;loc<leafsize;loc++)
      { 
        if (leafsID[loc]==OneID)  break;
      }
    }
    else if (AType==2)
    {
      l_prob = 0.0;
      OneNode = death_node(forest[NowTree], &l_prob);
      OneID=OneNode->nid;  
      Node_depth = depth(OneNode);
      leaf_prob = growth_prior(Node_depth, hypers);
      ll_before = 2*log(1.0 - growth_prior(Node_depth+1, hypers)) + log(leaf_prob);

      ll_after = log(1.0 - leaf_prob);
      p_forward = log(l_prob * 0.5 );
      p_backwards = log(1.0 / (leafsize-1) * ((leafsize==2)? 1.0 : 0.5));
      T_var=0;
      T_val=0;
      T_tau=0;


      for (loc=0;loc<leafsize;loc++)
      { 
        if (leafsID[loc]==2*OneID)  break;
      }
    }
    else if (AType==3)
    {
      std::vector<Node*> ngb = not_grand_branches(forest[NowTree]);
      OneNode = rand(ngb);
      OneID=OneNode->nid;  
      ll_before = 0;
      ll_after = 0;
      T_var = hypers.SampleVar();
      T_val=  OneNode->Getval(T_var);
      T_tau=RandExp(hypers.width);
      p_forward=0;
      p_backwards = 0;
      for (loc=0;loc<leafsize;loc++)
      { 
        if (leafsID[loc]==2*OneID)  break;
      }
    }

    leafsizeA=0;
    if (AType==1)
    {
      leafsizeA = leafsize+1;
    } 
    else if  (AType==2)
    {
      leafsizeA = leafsize-1;
    }    
    else if  (AType==3)
    {
      leafsizeA = leafsize;
    }  






    arma::vec w_i_1 = arma::zeros<arma::vec>(leafsize);
    arma::vec mu_hat_S_1 = arma::zeros<arma::vec>(leafsize);
    arma::mat Lambda_S_1 = arma::zeros<arma::mat>(leafsize, leafsize);

    arma::vec w_i_2 = arma::zeros<arma::vec>(leafsizeA);
    arma::vec mu_hat_S_2 = arma::zeros<arma::vec>(leafsizeA);
    arma::mat Lambda_S_2 = arma::zeros<arma::mat>(leafsizeA, leafsizeA);

    double TW;
    double LW;

    for(unsigned int i = 0; i < X.n_rows; i++) 
    {
      forest[NowTree]->GetW(X, i);
      for(int j = 0; j < leafsize; j++) 
      {
        w_i_1(j) = leafs[j]->current_weight;
      }

      if (AType==1)
      {
        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            LW=activation(X(i,T_var), T_val, forest[NowTree]->tau);
            //LW=activation(X(i,T_var), T_val, T_tau); 
            w_i_2(j)=w_i_1(j)*LW;
            w_i_2(j+1)=w_i_1(j)*(1-LW);
            j++;
          }
          else
          {
            w_i_2(j)=w_i_1(j-1);
          }
        }
         
      } 
      else if  (AType==2)
      {

        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            w_i_2(j)=w_i_1(j)+w_i_1(j+1);
          }
          else
          {
            w_i_2(j)=w_i_1(j+1);
          }
        }
      }    
      else if  (AType==3)
      {
        for(int j = 0; j < leafsizeA; j++) 
        { 
          if (j<loc)
          {
            w_i_2(j)=w_i_1(j);    
          }
          else if (loc==j)
          {
            TW=(forest[NowTree]->getptr(OneID))->current_weight;
            
            LW=activation(X(i,T_var), T_val, forest[NowTree]->tau);
            //LW=activation(X(i,T_var), T_val, T_tau); 
            w_i_2(j)=LW*TW;
            w_i_2(j+1)=TW*(1-LW);  
            j++;
          }
          else
          {
            w_i_2(j)=w_i_1(j);
          }
        }        
      }  
     
      mu_hat_S_1 = mu_hat_S_1 + res(i) * w_i_1;
      Lambda_S_1 = Lambda_S_1 + w_i_1 * arma::trans(w_i_1);

      mu_hat_S_2 = mu_hat_S_2 + res(i) * w_i_2;
      Lambda_S_2 = Lambda_S_2 + w_i_2 * arma::trans(w_i_2);

    }


 
 

    Lambda_S_1 = Lambda_S_1 / pow(hypers.sigma, 2) * hypers.temperature;
    mu_hat_S_1 = mu_hat_S_1 / pow(hypers.sigma, 2) * hypers.temperature;
    
    arma::mat Omega_inv1 = Lambda_S_1 + arma::eye(leafsize, leafsize) / pow(hypers.sigma_mu, 2);
    arma::vec mu_hat1 = solve(Omega_inv1, mu_hat_S_1);


    Lambda_S_2 = Lambda_S_2 / pow(hypers.sigma, 2) * hypers.temperature;
    mu_hat_S_2 = mu_hat_S_2 / pow(hypers.sigma, 2) * hypers.temperature;

    arma::mat Omega_inv2 = Lambda_S_2 + arma::eye(leafsizeA, leafsizeA) / pow(hypers.sigma_mu, 2);
    arma::vec mu_hat2 = solve(Omega_inv2, mu_hat_S_2);


    double val, sign;
    LogLT1 = -0.5 * leafsize * log(M_2_PI * pow(hypers.sigma_mu,2));
    log_det(val, sign, Omega_inv1 / M_2_PI);
    LogLT1 -= 0.5 * val;
    LogLT1 += 0.5 * dot(mu_hat1, Omega_inv1 * mu_hat1);

    LogLT2 = -0.5 * leafsizeA * log(M_2_PI * pow(hypers.sigma_mu,2));
    log_det(val, sign, Omega_inv2 / M_2_PI);
    LogLT2 -= 0.5 * val;
    LogLT2 += 0.5 * dot(mu_hat2, Omega_inv2 * mu_hat2);



    double log_trans_prob = ll_after + LogLT2 + p_backwards - ll_before - p_forward-LogLT1;

    if(log(arma::randu()) > log_trans_prob) 
    { 
      DType=0;
      LLogLT=LogLT1;  
      mu_samp = rmvnorm(mu_hat1, Omega_inv1); 
      for(int j = 0; j < leafsize; j++) 
      {
        MUlist[j]=mu_samp[j];
      }
      Fleafsize=leafsize;
      Pleafsize=Fleafsize+1;   
    }
    else
    {
      DType=1;
      LLogLT=LogLT2;
      mu_samp = rmvnorm(mu_hat2, Omega_inv2);      
      for(int j = 0; j < leafsizeA; j++) 
      {
        MUlist[j]=mu_samp[j];
      }
      Fleafsize=leafsizeA;
      Pleafsize=Fleafsize+1;
    }


    MPI_Allreduce(&Fleafsize, &Mleafsize, 1, MPI_INT, MPI_MAX,MPI_COMM_WORLD);







    int BlockSize=Mleafsize+10;


    if (DType)
    { 
      if (AType==1)
      {
        OneNode=forest[NowTree]->getptr(OneID);
        OneNode->AddLeaves();
        OneNode->var=T_var;
        OneNode->val=T_val;
        OneNode->GetLimits();


      }
      else if  (AType==2)
      {
        OneNode=forest[NowTree]->getptr(OneID);
        OneNode->DeleteLeaves();
      }
      else if (AType==3)
      {
        OneNode=forest[NowTree]->getptr(OneID);

        OneNode->var = T_var;
        OneNode->GetLimits();
        OneNode->val = T_val;
      }
    }




    if(opts.update_tau) forest[NowTree]->UpdateTauB(res, X, hypers,myrank,LLogLT,MUlist,leafsize+1);
 
    



    double * sendbuf;
    double * recvbuf;

    recvbuf = new double [(hypers.np+1)*BlockSize*sizeof(double)];
    sendbuf=  new double [BlockSize*sizeof(double)];
    sendbuf[0]=(double) AType;
    sendbuf[1]=(double) OneID;
    sendbuf[2]=(double) loc;
    sendbuf[3]=(double) T_var;
    sendbuf[4]=(double) T_val;
    sendbuf[5]=(double) T_tau;
    sendbuf[6]=(double) DType;
    sendbuf[7]=(double) Fleafsize;    
    sendbuf[8]=(double) forest[NowTree]->tau;
    sendbuf[9]=(double) NowTree;
    int Fsize=(int)sendbuf[7];
    for (int t3=0;t3<Fsize;t3++)
    {
      sendbuf[10+t3]=  MUlist[t3];
    }


    // srnstr.str("");
    // srnstr<<"Myrank"<<myrank<< "\n";
    // for (int t3=0;t3<BlockSize;t3++)
    // {
    //   srnstr <<sendbuf[t3]<<" ";
    // }
    // srnstr <<"\n";
    // cout << srnstr.str()<<std::flush;  





    MPI_Allgather(sendbuf, BlockSize, MPI_DOUBLE, recvbuf, BlockSize, MPI_DOUBLE, MPI_COMM_WORLD);
    
    //srnstr.str("");
    //srnstr<<"Myrank"<<myrank<< " " << BlockSize<<" " <<Fsize<<"Afore Barrier\n";
    //cout << srnstr.str()<<std::flush;  
    
    MPI_Barrier(MPI_COMM_WORLD);



    // srnstr.str("");
    // srnstr<<"Myrank"<<myrank<< "\n";
    // for (int t2=0;t2<=hypers.np;t2++)
    // { 
    //   srnstr <<"Tree ";
    //   for (int t3=0;t3<BlockSize;t3++)
    //   {
    //     srnstr <<recvbuf[BlockSize+t3]<<" ";
    //   }
    //   srnstr <<"\n";
    // }
  											
    // cout << srnstr.str()<<std::flush;  


   

     
    int upTree;
    for (int t2=0;t2<=hypers.np;t2++)
    { 

      // srnstr.str("");
      // srnstr<<"Myrank"<<myrank<< " " << t2<<" " <<"Start LOOP\n";
      // cout << srnstr.str()<<std::flush;  



         
      AType=(int)(recvbuf[t2*BlockSize]);
      OneID=(unsigned int)(recvbuf[t2*BlockSize+1]) ;
      loc=(int)(recvbuf[t2*BlockSize+2]);
      T_var=(int)(recvbuf[t2*BlockSize+3]);
      T_val=(double)(recvbuf[t2*BlockSize+4])  ;
      DType=(int) ( recvbuf[t2*BlockSize+6]) ;
      Fleafsize=(int)(recvbuf[t2*BlockSize+7])  ;    
      T_tau=recvbuf[t2*BlockSize+8];
      upTree=(int)(recvbuf[t2*BlockSize+9]);
      double * upmlist=new double [Fleafsize];
      for (int t3=0;t3<Fleafsize;t3++)
      {
        upmlist[t3]=recvbuf[t2*BlockSize+10+t3];
      }
      if (t2 != myrank)
      {
        if (DType)
        { 
          if (AType==1)
          {
            OneNode=forest[upTree]->getptr(OneID);
            OneNode->AddLeaves();
            OneNode->var=T_var;
            OneNode->val=T_val;
            OneNode->GetLimits();


          }
          else if  (AType==2)
          {
            OneNode=forest[upTree]->getptr(OneID);
            OneNode->DeleteLeaves();
          }
          else if (AType==3)
          {
            OneNode=forest[upTree]->getptr(OneID);

            OneNode->var = T_var;
            OneNode->GetLimits();
            OneNode->val = T_val;
          }
        }
      }  
      forest[upTree]->SetTau(T_tau);
      forest[upTree]->UpdateMuB(upmlist,myrank);
      delete [] upmlist;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // srnstr.str("");
    // srnstr<<"Myrank"<<myrank<<  "Start Block\n";
    // cout << srnstr.str()<<std::flush;  


    arma::vec Y_Block_E=arma::zeros<arma::vec>(Y.n_elem);
    for (int t2=0;t2<=hypers.np;t2++)
    { 
      arma::vec Y_predict=predict(forest[Tter[t1*(hypers.np +1)+t2]], X, hypers);
      Y_Block_E=Y_Block_E+Y_predict;
    }

    Y_hat = Y_hat -Y_Block_S+Y_Block_E;

    // srnstr.str("");
    // srnstr<<"Myrank"<<myrank<< " " <<t1<<"End Block\n";
    // cout << srnstr.str()<<std::flush;  


    delete [] recvbuf;
    delete [] sendbuf;

    delete [] MUlist;
  } 









}