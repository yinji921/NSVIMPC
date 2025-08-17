#include <curand_kernel.h>

#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s
#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM 8

#define TRACK_INFO_SIZE %(TRACK_INFO_SIZE)s
#define TRACK_WIDTH %(TRACK_WIDTH)s

#define m_Vehicle_m %(m_Vehicle_m)s
#define m_Vehicle_Iz %(m_Vehicle_Iz)s
#define m_Vehicle_lF %(m_Vehicle_lF)s
#define m_Vehicle_lR %(m_Vehicle_lR)s

#define m_Vehicle_IwF %(m_Vehicle_IwF)s
#define m_Vehicle_IwR %(m_Vehicle_IwR)s

#define m_Vehicle_lR %(m_Vehicle_lR)s
#define m_Vehicle_rF %(m_Vehicle_rF)s
#define m_Vehicle_rR %(m_Vehicle_rR)s
#define m_Vehicle_h %(m_Vehicle_h)s
#define m_Vehicle_tire_B %(m_Vehicle_tire_B)s
#define m_Vehicle_tire_C %(m_Vehicle_tire_C)s
#define m_Vehicle_tire_D %(m_Vehicle_tire_D)s

#define m_Vehicle_kSteering %(m_Vehicle_kSteering)s
#define m_Vehicle_cSteering %(m_Vehicle_cSteering)s

#define BUFFER_SIZE 16

#define m_g 9.80665
#define PI 3.14159265

__device__ curandState_t* curand_states[CURAND_KERNEL_N];

//NOTE potential performance increase by using __constant__ or similar
__device__ float Q[STATE_DIM*STATE_DIM];
__device__ float R[CONTROL_DIM*CONTROL_DIM];
__device__ float QN[STATE_DIM*STATE_DIM];

__device__ float track_points[TRACK_INFO_SIZE*2];
__device__ float track_mid_points[TRACK_INFO_SIZE*2];
__device__ float track_distance[TRACK_INFO_SIZE];
__device__ float goal_state[STATE_DIM];

extern "C" {
__global__ void init_curand_kernel(int seed){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= SAMPLE_COUNT*HORIZON*CONTROL_DIM) return;

  curandState_t* s = new curandState_t;
  if (s != 0) {
    curand_init(seed, id, 0, s);
  } else {
    printf("error initializing curand kernel\n");
  }

  curand_states[id] = s;
}

__global__ void set_cost_matrices(float* in_Q, float* in_R, float* in_QN){
  for (int i=0;i<STATE_DIM*STATE_DIM;i++){
    Q[i] = in_Q[i];
    QN[i] = in_QN[i];
  }
  for (int i=0;i<CONTROL_DIM*CONTROL_DIM;i++){
    R[i] = in_R[i];
  }
}

__global__ void set_track_info(float* in_track_points, float* in_track_distance){
  for (int i=0; i<TRACK_INFO_SIZE*2; i++){
    track_points[i] = in_track_points[i];
  }
  for (int i=0; i<TRACK_INFO_SIZE; i++){
    track_distance[i] = in_track_distance[i];
  }

  // find track midpoints
}

__global__ void set_goal_state(float* in_goal_state){
  for (int i=0; i<STATE_DIM; i++){
    goal_state[i] = in_goal_state[i];
  }
}

// mean: (m)
// cov : (m*m) covariance matrix
// values: (N * (H-1) * m) 
__global__ void generate_gaussian_noise(float* mean, float* cov, float *values){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  // failsafe, should never be true
  if (id >= CURAND_KERNEL_N) {return;}

  // NOTE: only the diagonal term in cov matrix is used
  float _cov[CONTROL_DIM*CONTROL_DIM];
  for (int i=0; i<CONTROL_DIM*CONTROL_DIM; i++){
    _cov[i] = cov[i];
  }

  float _mean[CONTROL_DIM];
  for (int i=0; i<CONTROL_DIM; i++){
    _mean[i] = mean[i];
  }

  curandState_t s = *curand_states[id];
  int start = id*(SAMPLE_COUNT*(HORIZON-1)*CONTROL_DIM)/CURAND_KERNEL_N;
  int end = min(SAMPLE_COUNT*(HORIZON-1)*CONTROL_DIM,(id+1)*(SAMPLE_COUNT*(HORIZON-1)*CONTROL_DIM)/CURAND_KERNEL_N);
  //printf("id %%d, %%d - %%d\n",id, start, end);

  for(int i=start; i < end; i+=CONTROL_DIM ) {
    for (int j=0; j<CONTROL_DIM; j++){
      float val = _mean[j] + curand_normal(&s) * _cov[j*CONTROL_DIM+j];
      values[i+j] = val;
    }
  }
  *curand_states[id] = s;
  /*
  if (id == 0){
    printf("start = %%d, end = %%d \n", start, end);
    printf("mean = %%.2f, %%.2f \n",_mean[0], _mean[1]);
    printf("cov = %%.2f, %%.2f \n",_cov[0], _cov[3]);
  }
  */
}

// calculate matrix multiplication
// A is assumed to be a stack of 2d matrix, offset instructs the index of 2d matrix to use
// out(m*p) = A_offset(m*n) @ B(n*p), A matrix start from A+offset*n*m
__device__
void matrix_multiply_helper( float* A, int offset, float* B, int m, int n, int p, float* out){
  for (int i=0; i<m; i++){
    for (int j=0; j<p; j++){
      out[i*p + j] = 0;
      for (int k=0; k<n; k++){
        out[i*p + j] += A[offset*n*m + i*n + k] * B[k*p + j];
      }
    }
  }
}

// instead of rewriting out_add, add to it
__device__
void matrix_multiply_helper_add( float* A, int offset, float* B, int m, int n, int p, float* out_add){
  for (int i=0; i<m; i++){
    for (int j=0; j<p; j++){
      for (int k=0; k<n; k++){
        out_add[i*p + j] += A[offset*n*m + i*n + k] * B[k*p + j];
      }
    }
  }
}

__device__
void matrix_multiply_scalar(float* A, float coeff, int m, int n){
  for (int i=0; i<m; i++){
    for (int j=0; j<n; j++){
      A[i*n + j] *= coeff;
    }
  }

}

// x: n*1, A: n*n, y:n*1
__device__ void triple_product_xTAy(float* x,float* A, float* y, int n, float* result){
  float buffer[BUFFER_SIZE];
  matrix_multiply_helper(x,0,A,1,n,n,buffer);
  matrix_multiply_helper(buffer,0,y,1,n,1,result);
}
// x0: n

// ref_control: cpu: m*(H-1) gpu:(H-1)*m
// control_noise: N*(H-1)*m
// trajectories: cpu:N*n*H gpu: N*H*n
__global__ void propagate_dynamics(float *x0,float* ref_control, float* control_noise, float* trajectories,int num_controlled_trajectories, float* Ks, float* As, float* Bs){
  int H = HORIZON;
  int n = STATE_DIM;
  int m = CONTROL_DIM;
  float dt = 0.01;
  float y[STATE_DIM] = {0.0};
  float temp2[CONTROL_DIM] = {0.0};
  //float temp8[STATE_DIM] = {0.0};

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= SAMPLE_COUNT) {return;}
  float vx,vy,wz,wF,wR;
  float psi,X,Y;
  vx = x0[0];
  vy = x0[1];
  wz = x0[2];
  wF = x0[3];
  wR = x0[4];
  // cartesian / autorally_dynamics_map
  psi = x0[5];
  X = x0[6];
  Y = x0[7];

  trajectories[id*(H*n) + 0] = vx;
  trajectories[id*(H*n) + 1] = vy;
  trajectories[id*(H*n) + 2] = wz;
  trajectories[id*(H*n) + 3] = wF;
  trajectories[id*(H*n) + 4] = wR;
  trajectories[id*(H*n) + 5] = psi;
  trajectories[id*(H*n) + 6] = X;
  trajectories[id*(H*n) + 7] = Y;

  //u: steering, throttle(>0)
  // perturbed control[0..H-2]: control_noise[id*(H-1)*m+t*(m)+0/1]+ref_control[t*m+0/1]
  // trajectories[0..H-1]: trajectories[id*(H*n) + t*n + this_n]
  for (int t=0;t<(H-1);t++){
    float steering = control_noise[id*(H-1)*m+t*(m)+0];
    float throttle = control_noise[id*(H-1)*m+t*(m)+1];
    if (id < num_controlled_trajectories){
      steering += ref_control[t*m+0];
      throttle += ref_control[t*m+1];
      //u += Ky
      // K: 2*8, y:8*1
      matrix_multiply_helper(Ks,t,y,CONTROL_DIM,STATE_DIM,1,temp2);
      steering += temp2[0];
      throttle += temp2[1];
      // y+ = Ay + Bw w=noise
      matrix_multiply_helper_add(As,t,y,STATE_DIM,STATE_DIM,1,y);
      matrix_multiply_helper_add(Bs,t,control_noise+id*(H-1)*m+t*(m),STATE_DIM,CONTROL_DIM,1,y);
    }
    float delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering;
    throttle = throttle>0?throttle:0;
    /*
    if (id==50){
      printf("[gpu] id51 steering=%%.2f, delta=%%.2f, throttle=%%.2f\n",steering,delta,throttle);
    }
    */

    throttle = max(throttle,0.0f);
    // internal step time 0.01s, step time 0.05s
    for (int i=0; i<5; i++){
      float beta = atan2f(vy,vx);
      float V = sqrtf(vx*vx+vy*vy);
      /*
      if (id==50){
        printf("[GPU] id51, step %%d, beta = %%.3f, V = %%.3f, wz=%%.3f\n",i,beta,V,wz);
      }
      */
      float vFx = V * cosf(beta - delta) + wz * m_Vehicle_lF * sinf(delta);
      float vFy = V * sinf(beta - delta) + wz * m_Vehicle_lF * cosf(delta);
      float vRx = vx;
      float vRy = vy - wz * m_Vehicle_lR;

      /*
      if (id==50){
        printf("[GPU] id51, step %%d, vFx=%%.3f, vFy=%%.3f, vRx=%%.3f, vRy=%%.3f\n",i,vFx,vFy,vRx,vRy);
      } // mismatch starting 3rd step
      */

      // FIXME not sure if totally right
      wF = abs(wF)<0.1?0.1:wF;
      wR = abs(wR)<0.1?0.1:wR;

      float sFx = (wF > 0)? ((vFx - wF * m_Vehicle_rF) / (wF * m_Vehicle_rF)):0;
      float sRx = (wR > 0)? ((vRx - wR * m_Vehicle_rR) / (wR * m_Vehicle_rR)):0;
      float sFy = (vFx > 0)? (vFy / (wF * m_Vehicle_rF)): 0;
      float sRy = (vRx > 0)? (vRy / (wR * m_Vehicle_rR)): 0;

      /*
      if (id==50){
        printf("[GPU] id51, step %%d, wR=%%.3f, wF=%%.3f,sFx=%%.3f, sFy=%%.3f, sRx=%%.3f, sRy=%%.3f\n",i,wR,wF,sFx,sFy,sRx,sRy);
      } // mismatch wR step 1, mismatch at step 2
      */

      float sF = sqrtf(sFx * sFx + sFy * sFy) + 1e-2;
      float sR = sqrtf(sRx * sRx + sRy * sRy) + 1e-2;

      float muF = m_Vehicle_tire_D * sinf(m_Vehicle_tire_C * atanf(m_Vehicle_tire_B * sF));
      float muR = m_Vehicle_tire_D * sinf(m_Vehicle_tire_C * atanf(m_Vehicle_tire_B * sR));
      /*
      if (id==50){
        printf("[GPU] id51, step %%d, sF=%%.3f, sR=%%.3f, muF=%%.3f, muR=%%.3f\n",i,sF,sR,muF,muR);
      } //mismatch sR step 1
      */

      float muFx = -sFx / sF * muF;
      float muFy = -sFy / sF * muF;
      float muRx = -sRx / sR * muR;
      float muRy = -sRy / sR * muR;
      /*
      if (id==50){
        printf("[GPU] id51, step %%d, muFx=%%.3f, muFy=%%.3f, muRx=%%.3f, muRy=%%.3f\n",i,muFx,muFy,muRx,muRy);
      } // muRx step 1
      */


      float fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (
              m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * cosf(delta) - muFy * sinf(delta) - muRx));
      float fRz = m_Vehicle_m * m_g - fFz;

      float fFx = fFz * muFx;
      float fRx = fRz * muRx;
      float fFy = fFz * muFy;
      float fRy = fRz * muRy;
      /*
      if (id==50){
        printf("[GPU] id51, step %%d, fFz=%%.3f, fRz=%%.3f,fFx=%%.3f, fFy=%%.3f, fRx=%%.3f, fRy=%%.3f\n",i,fFz,fRz,fFx,fFy,fRx,fRy);
      } // step 1
      */

      float ax = ((fFx * cosf(delta) - fFy * sinf(delta) + fRx) / m_Vehicle_m + vy * wz);
      vx = vx + dt * (
                  (fFx * cosf(delta) - fFy * sinf(delta) + fRx) / m_Vehicle_m + vy * wz);
      vy = vy + dt * ((fFx * sinf(delta) + fFy * cosf(delta) + fRy) / m_Vehicle_m - vx * wz);
      wz = wz + dt * (
                        (fFy * cosf(delta) + fFx * sinf(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR) / m_Vehicle_Iz;
      wF = wF - dt * m_Vehicle_rF / m_Vehicle_IwF * fFx;
      /*
      if (id==50){
        printf("[GPU] id51, step %%d, ax=%%.3f, vx+=%%.3f,vy+=%%.3f,wz+=%%.3f,wF+=%%.3f\n",i,ax,vx,vy,wz,wF);
      }
      */

      float dt_wR = atanf(-(throttle -5 + (wR-30)*0.06)*4) * 90/PI + 57 -0.00625*expf(-(-10.00000 + 33.33333*(throttle-0.20000)));
      dt_wR = dt_wR<-50?-50:dt_wR;
      wR = wR + dt * dt_wR;

      /*
      if (id==50){
        printf("[GPU] id51, step %%d, wR=%%.3f, throttle=%%.3f,dt_wR=%%.3f, wR+=%%.3f\n",i,wR,throttle,dt_wR,wR);
      }
      */

      psi = psi + dt * wz;
      X = X + dt * (cosf(psi) * vx - sinf(psi) * vy);
      Y = Y + dt * (sinf(psi) * vx + cosf(psi) * vy);
    }

    // FIXME may be incorrect
    wR = wR<5?5:wR;
    wF = wF<5?5:wF;
    trajectories[id*(H*n) + (t+1)*n + 0] = vx;
    trajectories[id*(H*n) + (t+1)*n + 1] = vy;
    trajectories[id*(H*n) + (t+1)*n + 2] = wz;
    trajectories[id*(H*n) + (t+1)*n + 3] = wF;
    trajectories[id*(H*n) + (t+1)*n + 4] = wR;
    trajectories[id*(H*n) + (t+1)*n + 5] = psi;
    trajectories[id*(H*n) + (t+1)*n + 6] = X;
    trajectories[id*(H*n) + (t+1)*n + 7] = Y;

  }
}

__device__ void cartesian_to_curvilinear_old(float* in_state, float* out_state, int* guess){
  // find closest reference point
  int N = TRACK_INFO_SIZE;
  float min_dist_sqr = 100000.0;
  int min_index = 0;
  // start/finish index for searching
  int start = 0;
  int end = 0;

  if ((*guess)<0){
    start = 0;
    end = TRACK_INFO_SIZE;
  } else {
    // TODO need more tuning
    start = (*guess)-10;
    end = (*guess)+20;
  }

  for(int i=start;i<end;i++){
    // 5:psi, 6,7:X,Y
    float dx = track_points[(i%%N)*2+0] - in_state[6];
    float dy = track_points[(i%%N)*2+1] - in_state[7];
    float dist_sqr = dx*dx+dy*dy;
    if (dist_sqr<min_dist_sqr){
      min_dist_sqr = dist_sqr;
      min_index = i%%N;
    }
  }
  *guess = min_index;

  // find lateral and longitudinal distance from curve
  // track tangent direction normalize(p_1-p_{-1})
  float dir[2];
  dir[0] = track_points[((min_index+1)%%N)*2+0] - track_points[((min_index-1)%%N)*2+0];
  dir[1] = track_points[((min_index+1)%%N)*2+1] - track_points[((min_index-1)%%N)*2+1];
  //dir[0] = track_points[((min_index+1)%%N)*2+0] - track_points[((min_index)%%N)*2+0];
  //dir[1] = track_points[((min_index+1)%%N)*2+1] - track_points[((min_index)%%N)*2+1];
  float dir_norm = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]);
  // vector: p0 -> car
  float vec[2];
  vec[0] = in_state[6] - track_points[min_index*2+0];
  vec[1] = in_state[7] - track_points[min_index*2+1];
  // cross track error
  // left positive
  // = cross(dir,vec)
  float e_y = dir[0]*vec[1]-dir[1]*vec[0];
  // longitudinal distance
  // = distance(ref_point)+dot(dir,vec)
  // TODO wrap distance around for end of index
  float s = track_distance[min_index]+dir[0]*vec[0]+dir[1]*vec[1];
  // heading error = heading - ref heading
  // TODO wrap angle to [-pi,pi]
  float e_heading = in_state[5] - atan2f(dir[1],dir[0]);
  e_heading = fmodf(e_heading+3*PI,2*PI) - PI;

  out_state[0] = in_state[0];
  out_state[1] = in_state[1];
  out_state[2] = in_state[2];
  out_state[3] = in_state[3];
  out_state[4] = in_state[4];
  out_state[5] = e_heading;
  out_state[6] = e_y;
  out_state[7] = s;
}

__device__ void cartesian_to_curvilinear(float* in_state, float* out_state, int* guess){
  // find closest reference point
  int N = TRACK_INFO_SIZE;
  float min_dist_sqr = 100000.0;
  int min_index = 0;
  // start/finish index for searching
  int start = 0;
  int end = 0;

  if ((*guess)<0){
    start = 0;
    end = TRACK_INFO_SIZE;
  } else {
    // TODO need more tuning
    start = (*guess)-10;
    end = (*guess)+20;
  }

  for(int i=start;i<end;i++){
    // 5:psi, 6,7:X,Y
    float dx = track_points[(i%%N)*2+0] - in_state[6];
    float dy = track_points[(i%%N)*2+1] - in_state[7];
    float dist_sqr = dx*dx+dy*dy;
    if (dist_sqr<min_dist_sqr){
      min_dist_sqr = dist_sqr;
      min_index = i%%N;
    }
  }
  *guess = min_index;

  // find lateral and longitudinal distance from curve
  // track tangent direction normalize(p_1-p_{-1})
  float dir[2];
  dir[0] = track_points[((min_index+1)%%N)*2+0] - track_points[((min_index-1)%%N)*2+0];
  dir[1] = track_points[((min_index+1)%%N)*2+1] - track_points[((min_index-1)%%N)*2+1];
  //dir[0] = track_points[((min_index+1)%%N)*2+0] - track_points[((min_index)%%N)*2+0];
  //dir[1] = track_points[((min_index+1)%%N)*2+1] - track_points[((min_index)%%N)*2+1];
  float dir_norm = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]);
  // vector: p0 -> car
  float vec[2];
  vec[0] = in_state[6] - track_points[min_index*2+0];
  vec[1] = in_state[7] - track_points[min_index*2+1];
  // cross track error
  // left positive
  // = cross(dir,vec)
  float e_y = dir[0]*vec[1]-dir[1]*vec[0];
  // longitudinal distance
  // = distance(ref_point)+dot(dir,vec)
  // TODO wrap distance around for end of index
  float s = track_distance[min_index]+dir[0]*vec[0]+dir[1]*vec[1];
  // heading error = heading - ref heading
  // TODO wrap angle to [-pi,pi]
  float e_heading = in_state[5] - atan2f(dir[1],dir[0]);
  e_heading = fmodf(fmodf(e_heading+4*PI,2*PI)+PI,2*PI) - PI;
  if (e_heading>PI || e_heading<-PI){
    printf("%%.3f\n",in_state[5] - atan2f(dir[1],dir[0]));
  }

  out_state[0] = in_state[0];
  out_state[1] = in_state[1];
  out_state[2] = in_state[2];
  out_state[3] = in_state[3];
  out_state[4] = in_state[4];
  out_state[5] = e_heading;
  out_state[6] = e_y;
  out_state[7] = s;
}


// subtract goal state from given state, done in place
__device__ void subtract_goal_state(float* state){
  for (int i=0;i<STATE_DIM;i++){
    state[i] = state[i] - goal_state[i];
  }
}

// evaluate cost for an entire trajectory
// trajectory dim: num_of_traj, horizon, state_dim
__global__ void evaluate_trajectory_cost(float* trajectory, float* ref_control, float* control_noise, float* cost_vec, int num_controlled_trajectories){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // convert to map state
  float map_state[STATE_DIM];
  int H = HORIZON;
  int m = CONTROL_DIM;
  int n = STATE_DIM;

  int guess = -1;
  float cost = 0;
  // add step cost
  for (int i=0;i<HORIZON-1;i++){
    cartesian_to_curvilinear(trajectory+id*(HORIZON*STATE_DIM)+i*(STATE_DIM), map_state, &guess);
    /*
    if(id==50 && i==HORIZON-2){
      printf("[gpu] step %%d, index = %%d, e_psi=%%.3f, e_y=%%.3f, s=%%.3f\n",i,guess,map_state[5],map_state[6],map_state[7]);
    }
    */
    if(id==10000){
      printf("");
    }
    float step_cost = 0;
    float temp = 0;
    float u[CONTROL_DIM];
    float v[CONTROL_DIM];
    // epsilon
    float e[CONTROL_DIM];
    if (id < num_controlled_trajectories){
      u[0] = control_noise[id*(H-1)*m+i*(m)+0]+ref_control[i*m+0];
      u[1] = control_noise[id*(H-1)*m+i*(m)+1]+ref_control[i*m+1];
      v[0] = ref_control[i*m+0];
      v[1] = ref_control[i*m+1];
    } else {
      u[0] = control_noise[id*(H-1)*m+i*(m)+0];
      u[1] = control_noise[id*(H-1)*m+i*(m)+1];
      v[0] = 0.0;
      v[1] = 0.0;
    }
    e[0] = control_noise[id*(H-1)*m+i*(m)+0];
    e[1] = control_noise[id*(H-1)*m+i*(m)+1];
    // step cost = 1/2*xTQx + 1/2*uTRu + vTRe + 1/2*eRe
    /*
    if(id==50){
      printf("[gpu] u0=%%.3f,u0=%%.3f ",u[0],u[1]);
    }
    */

    subtract_goal_state(map_state);
    triple_product_xTAy(map_state,Q,map_state,n,&temp);
    step_cost += 0.5*temp;
    // removing this print increses num of registers used
    /*
    if(id==50){
      printf("[gpu] step=%%d, s=%%.3f, 1/2xQx = %%.3f ",i,map_state[7],0.5*temp);
    }
    */

    triple_product_xTAy(u,R,u,m,&temp);
    step_cost += 0.5*temp;
    /*
    if(id==50){
      printf(" 1/2uRu = %%.3f ",0.5*temp);
    }
    */
    triple_product_xTAy(v,R,e,m,&temp);
    step_cost += temp;
    /*
    if(id==50){
      printf(" vRe = %%.3f ",temp);
    }
    */

    triple_product_xTAy(e,R,e,m,&temp);
    step_cost += 0.5*temp;
    /*
    if(id==50){
      printf(" 1/2eRe = %%.3f ",0.5*temp);
    }
    */
    // track boundary cost
    step_cost += 10.0*powf(abs(map_state[6]) - (TRACK_WIDTH - 0.3), 2.0);

    cost += step_cost;

    /*
    if(id==50){
      printf(" step_cost=%%.3f, total = %%.3f\n",step_cost,cost);
    }
    */
  }
  // add terminal cost = 1/2*xTQx
  float temp = 0;
  cartesian_to_curvilinear(trajectory+id*(HORIZON*STATE_DIM)+(HORIZON-1)*(STATE_DIM), map_state, &guess);
  subtract_goal_state(map_state);
  triple_product_xTAy(map_state,QN,map_state,n,&temp);
  cost += 0.5*temp;
  cost_vec[id] = cost;
  /*
  if (id==50){
    printf("gpu: %%d, %%.2f, %%.2f, %%.2f, terminal cost %%.2f\n",guess, map_state[5],map_state[6],map_state[7],0.5*temp);
    //printf("gpu: QN[0,0] %%.2f\n",QN[0]);
  }
  */
}

// extern "C"
}

