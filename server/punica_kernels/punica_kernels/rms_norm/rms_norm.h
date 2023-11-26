template <typename T>
bool rms_norm(T *__restrict__ output, const T *__restrict__ input,
              const T *__restrict__ weight, int rows, int columns,
              float epsilon);
