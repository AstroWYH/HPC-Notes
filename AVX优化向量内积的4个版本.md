```cpp
#include <immintrin.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;

#define START_LOG_TIME()                                  \
  std::chrono::high_resolution_clock::time_point t1 =     \
      std::chrono::high_resolution_clock::now();          \
  std::chrono::high_resolution_clock::time_point t2 = t1; \
  std::chrono::duration<double> time_used =               \
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
#define PRINT_COST_TIME(name)                                             \
  t2 = std::chrono::high_resolution_clock::now();                         \
  time_used =                                                             \
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); \
  std::cout << std::fixed << name                                         \
            << " TIME COST: " << (time_used.count() * 1000) << " ms."     \
            << std::endl;                                                 \
  t1 = t2;

template <typename T>
inline T i_plane_xxx_norm_naive(const T pi[4], const T p[3]) {
  return pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3];
}

template <typename T>
inline T i_plane_xxx_norm_opt_a(const T pi[4], const T p[3]) {
  __m128 pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
  __m128 p_sse = _mm_set_ps(p[0], p[1], p[2], 1.0);
  pi_sse = _mm_mul_ps(pi_sse, p_sse);
  return pi_sse[0] + pi_sse[1] + pi_sse[2] + pi_sse[3];
}

template <typename T>
inline T i_plane_xxx_norm_opt_b(const T pi[4], const T p[3]) {
  __m128 pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
  __m128 p_sse = _mm_set_ps(p[0], p[1], p[2], 1.0);
  pi_sse = _mm_mul_ps(pi_sse, p_sse);
  pi_sse = _mm_hadd_ps(pi_sse, pi_sse);
  pi_sse = _mm_hadd_ps(pi_sse, pi_sse);
  return _mm_cvtss_f32(pi_sse);
}

template <typename T>
inline T i_plane_xxx_norm_opt_c(const T pi[4], const T p[3]) {
  __m128 pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
  __m128 p_sse = _mm_set_ps(p[0], p[1], p[2], 1.0);
  p_sse = _mm_dp_ps(pi_sse, p_sse, 0b11110001);
  return _mm_cvtss_f32(p_sse);
}

bool is_first_d = true;
template <typename T>
inline T i_plane_xxx_norm_opt_d(const T pi[4], const T p[3]) {
  static __m128 pi_sse;
  if (is_first_d) {
    pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
    is_first_d = false;
  }
  __m128 p_sse = _mm_set_ps(p[0], p[1], p[2], 1.0);
  p_sse = _mm_dp_ps(pi_sse, p_sse, 0b11110001);
  return _mm_cvtss_f32(p_sse);
}

bool is_first_e = true;
template <typename T>
inline T i_plane_xxx_norm_opt_e(const T pi[4], const T p[3]) {
  static __m128 pi_sse;
  if (is_first_e) {
    pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
    is_first_e = false;
  }
  return _mm_cvtss_f32(
      _mm_dp_ps(pi_sse, _mm_set_ps(p[0], p[1], p[2], 1.0), 0b11110001));
}

vector<float> vec_tmp_naive_cpp(1);
template <typename T>
inline T i_plane_xxx_norm_naive_cpp(const T pi[4], const T p[3]) {
  float res = pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3];
  /***********cpp*********/
  for (int i = 0; i < 1; i++) {
    float ans = (pi[0] * pi[1] + pi[2] - pi[3]);
    vec_tmp_naive_cpp[i] = ans;
  }
  /***********cpp*********/
  return res;
}

vector<float> vec_tmp_opt_c_cpp(1);
template <typename T>
inline T i_plane_xxx_norm_opt_c_cpp(const T pi[4], const T p[3]) {
  __m128 pi_sse = _mm_set_ps(pi[0], pi[1], pi[2], pi[3]);
  __m128 p_sse = _mm_set_ps(p[0], p[1], p[2], 1.0);
  p_sse = _mm_dp_ps(pi_sse, p_sse, 0b11110001);
  /***********cpp*********/
  for (int i = 0; i < 1; i++) {
    float ans = (pi[0] * pi[1] + pi[2] - pi[3]);
    vec_tmp_naive_cpp[i] = ans;
  }
  /***********cpp*********/
  return _mm_cvtss_f32(p_sse);
}

int main() {
  long loop_count = 100000;
  // float pi[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // 替换为实际的平面参数
  // float p[3] = {5.0f, 6.0f, 7.0f};         // 替换为实际的点坐标

  float distance_naive = 0.0f;
  float distance_opt_a = 0.0f;
  float distance_opt_b = 0.0f;
  float distance_opt_c = 0.0f;
  float distance_opt_d = 0.0f;
  float distance_opt_e = 0.0f;
  float distance_naive_cpp = 0.0f;
  float distance_opt_c_cpp = 0.0f;

  vector<float> vec_distance_naive;
  vector<float> vec_distance_opt_a;
  vector<float> vec_distance_opt_b;
  vector<float> vec_distance_opt_c;
  vector<float> vec_distance_opt_d;
  vector<float> vec_distance_opt_e;
  vector<float> vec_distance_naive_cpp;
  vector<float> vec_distance_opt_c_cpp;

  vec_distance_naive.resize(loop_count);
  vec_distance_opt_a.resize(loop_count);
  vec_distance_opt_b.resize(loop_count);
  vec_distance_opt_c.resize(loop_count);
  vec_distance_opt_d.resize(loop_count);
  vec_distance_opt_e.resize(loop_count);
  vec_distance_naive_cpp.resize(loop_count);
  vec_distance_opt_c_cpp.resize(loop_count);

  int cnt = 1;
  while (cnt--) {
    std::ofstream outFile("random_values.txt");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    for (int i = 0; i < loop_count; i++) {
      float random_params[4] = {dis(gen), dis(gen), dis(gen), dis(gen)};
      float random_p[3] = {dis(gen), dis(gen), dis(gen)};

      outFile << random_params[0] << " " << random_params[1] << " "
              << random_params[2] << " " << random_params[3] << " "
              << random_p[0] << " " << random_p[1] << " " << random_p[2]
              << "\n";
    }
    outFile.close();

    std::ifstream inFile("random_values.txt");
    std::vector<float> random_values(loop_count * 7);

    for (int i = 0; i < loop_count * 7; i++) {
      inFile >> random_values[i];
    }
    inFile.close();

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_naive = i_plane_xxx_norm_naive(params, point);
        vec_distance_naive[i] = distance_naive;
      }
      PRINT_COST_TIME("naive")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_a = i_plane_xxx_norm_opt_a(params, point);
        vec_distance_opt_a[i] = distance_opt_a;
      }
      PRINT_COST_TIME("opt_a")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_b = i_plane_xxx_norm_opt_b(params, point);
        vec_distance_opt_b[i] = distance_opt_b;
      }
      PRINT_COST_TIME("opt_b")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_c = i_plane_xxx_norm_opt_c(params, point);
        vec_distance_opt_c[i] = distance_opt_c;
      }
      PRINT_COST_TIME("opt_c")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_d = i_plane_xxx_norm_opt_d(params, point);
        vec_distance_opt_d[i] = distance_opt_d;
      }
      PRINT_COST_TIME("opt_d")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_e = i_plane_xxx_norm_opt_e(params, point);
        vec_distance_opt_e[i] = distance_opt_e;
      }
      PRINT_COST_TIME("opt_e")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_naive_cpp = i_plane_xxx_norm_naive_cpp(params, point);
        vec_distance_naive_cpp[i] = distance_naive_cpp;
      }
      PRINT_COST_TIME("naive_cpp")
    }

    {
      START_LOG_TIME()
      for (int i = 0; i < loop_count; i++) {
        float* params = &random_values[i * 7];
        float* point = &random_values[i * 7 + 4];
        distance_opt_c_cpp = i_plane_xxx_norm_opt_c_cpp(params, point);
        vec_distance_opt_c_cpp[i] = distance_opt_c_cpp;
      }
      PRINT_COST_TIME("opt_c_cpp")
    }
  }

  std::cout << "naive distance: " << distance_naive << std::endl;
  std::cout << "opt_a distance: " << distance_opt_a << std::endl;
  std::cout << "opt_b distance: " << distance_opt_b << std::endl;
  std::cout << "opt_c distance: " << distance_opt_c << std::endl;
  std::cout << "opt_d distance: " << distance_opt_d << std::endl;
  std::cout << "opt_e distance: " << distance_opt_e << std::endl;
  std::cout << "naive_cpp distance: " << distance_naive_cpp << std::endl;
  std::cout << "opt_c_cpp distance: " << distance_opt_c_cpp << std::endl;

#ifdef __AVX2__
  cout << "avx define " << endl;
#endif

  return 0;
}
```
