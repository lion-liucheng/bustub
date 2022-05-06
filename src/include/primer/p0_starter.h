//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// p0_starter.h
//
// Identification: src/include/primer/p0_starter.h
//
// Copyright (c) 2015-2020, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "common/exception.h"

namespace bustub {

/**
 * The Matrix type defines a common
 * interface for matrix operations.
 */
template <typename T>
class Matrix {
 protected:
  /**
   * TODO(P0): Add implementation
   *
   * Construct a new Matrix instance.
   * @param rows The number of rows
   * @param cols The number of columns
   *
   */
  Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
    linear_ = new T[rows * cols];
    memset(linear_, 0, sizeof(T) * rows_ * cols_);
  }

  /** The number of rows in the matrix */
  int rows_;
  /** The number of columns in the matrix */
  int cols_;

  /**
   * TODO(P0): Allocate the array in the constructor.
   * TODO(P0): Deallocate the array in the destructor.
   * A flattened array containing the elements of the matrix.
   */
  T *linear_;

 public:
  /** @return The number of rows in the matrix */
  virtual int GetRowCount() const = 0;

  /** @return The number of columns in the matrix */
  virtual int GetColumnCount() const = 0;

  /**
   * Get the (i,j)th matrix element.
   *
   * Throw OUT_OF_RANGE if either index is out of range.
   *
   * @param i The row index
   * @param j The column index
   * @return The (i,j)th matrix element
   * @throws OUT_OF_RANGE if either index is out of range
   */
  virtual T GetElement(int i, int j) const = 0;

  /**
   * Set the (i,j)th matrix element.
   *
   * Throw OUT_OF_RANGE if either index is out of range.
   *
   * @param i The row index
   * @param j The column index
   * @param val The value to insert
   * @throws OUT_OF_RANGE if either index is out of range
   */
  virtual void SetElement(int i, int j, T val) = 0;

  /**
   * Fill the elements of the matrix from `source`.
   *
   * Throw OUT_OF_RANGE in the event that `source`
   * does not contain the required number of elements.
   *
   * @param source The source container
   * @throws OUT_OF_RANGE if `source` is incorrect size
   */
  virtual void FillFrom(const std::vector<T> &source) = 0;

  /**
   * Destroy a matrix instance.
   * TODO(P0): Add implementation
   */
  virtual ~Matrix() { delete[] linear_; }
};

/**
 * The RowMatrix type is a concrete matrix implementation.
 * It implements the interface defined by the Matrix type.
 */
template <typename T>
class RowMatrix : public Matrix<T> {
 public:
  /**
   * TODO(P0): Add implementation
   *
   * Construct a new RowMatrix instance.
   * @param rows The number of rows
   * @param cols The number of columns
   */
  RowMatrix(int rows, int cols) : Matrix<T>(rows, cols) {
    data_ = new T *[rows];
    for (int i = 0; i < rows; i++) {
      // data_[i] = linear_[i * cols]; //这里会找不到linear，不太清楚为啥会这样？？
      // data_[i] = this->linear_[i * cols]; 后面这个是值，对指针运算应该直接加
      data_[i] = this->linear_ + i * cols;
    }
  }

  /**
   * TODO(P0): Add implementation
   * @return The number of rows in the matrix
   */
  int GetRowCount() const override { return this->rows_; }

  /**
   * TODO(P0): Add implementation
   * @return The number of columns in the matrix
   */
  int GetColumnCount() const override { return this->cols_; }

  /**
   * TODO(P0): Add implementation
   *
   * Get the (i,j)th matrix element.
   *
   * Throw OUT_OF_RANGE if either index is out of range.
   *
   * @param i The row index
   * @param j The column index
   * @return The (i,j)th matrix element
   * @throws OUT_OF_RANGE if either index is out of range
   */
  T GetElement(int i, int j) const override {
    //这里需要做好鲁邦，否则测试时会挂掉
    if (i >= this->rows_ || i < 0 || j >= this->cols_ || j < 0) {
      throw Exception(ExceptionType::OUT_OF_RANGE, "OUT_OF_RANGE");
    }
    return data_[i][j];  //指针当做数组
  }

  /**
   * Set the (i,j)th matrix element.
   *
   * Throw OUT_OF_RANGE if either index is out of range.
   *
   * @param i The row index
   * @param j The column index
   * @param val The value to insert
   * @throws OUT_OF_RANGE if either index is out of range
   */
  void SetElement(int i, int j, T val) override {
    if (i >= this->rows_ || i < 0 || j >= this->cols_ || j < 0) {
      throw Exception(ExceptionType::OUT_OF_RANGE, "OUT_OF_RANGE");
    }
    data_[i][j] = val;
  }

  /**
   * TODO(P0): Add implementation
   *
   * Fill the elements of the matrix from `source`.
   *
   * Throw OUT_OF_RANGE in the event that `source`
   * does not contain the required number of elements.
   *
   * @param source The source container
   * @throws OUT_OF_RANGE if `source` is incorrect size
   */
  void FillFrom(const std::vector<T> &source) override {
    int source_size = static_cast<int>(source.size());
    // source_size变量风格
    // 这样符合C++新标准
    if (source_size != this->cols_ * this->rows_) {
      // 这里竟然会对size转化成int才行？ 有点离谱，需要详细看一下
      throw Exception(ExceptionType::OUT_OF_RANGE, "OUT_OF_RANGE");
    }
    for (int i = 0; i < this->cols_ * this->rows_; i++) {
      this->linear_[i] = source[i];
    }
  }

  /**
   * TODO(P0): Add implementation
   *
   * Destroy a RowMatrix instance.
   */
  ~RowMatrix() final {
    // 后面没有继承，加上final
    // 继承了基类的析构函数删除linea
    delete[] data_;  // 删除自己的部分
  }

 private:
  /**
   * A 2D array containing the elements of the matrix in row-major format.
   *
   * TODO(P0):
   * - Allocate the array of row pointers in the constructor.
   * - Use these pointers to point to corresponding elements of the `linear` array.
   * - Don't forget to deallocate the array in the destructor.
   */
  T **data_;
};

/**
 * The RowMatrixOperations class defines operations
 * that may be performed on instances of `RowMatrix`.
 */
template <typename T>
class RowMatrixOperations {
 public:
  /**
   * Compute (`matrixA` + `matrixB`) and return the result.
   * Return `nullptr` if dimensions mismatch for input matrices.
   * @param matrixA Input matrix
   * @param matrixB Input matrix
   * @return The result of matrix addition
   */
  static std::unique_ptr<RowMatrix<T>> Add(const RowMatrix<T> *matrixA, const RowMatrix<T> *matrixB) {
    // TODO(P0): Add implementation
    if (matrixA->GetRowCount() != matrixB->GetRowCount() || matrixA->GetColumnCount() != matrixB->GetColumnCount()) {
      return std::unique_ptr<RowMatrix<T>>(nullptr);
    }
    int row = matrixA->GetRowCount();
    int col = matrixA->GetColumnCount();
    std::unique_ptr<RowMatrix<T>> matrix_c = std::make_unique<RowMatrix<T>>(row, col);
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        matrix_c->SetElement(i, j, matrixA->GetElement(i, j) + matrixB->GetElement(i, j));
      }
    }

    return matrix_c;
  }

  /**
   * Compute the matrix multiplication (`matrixA` * `matrixB` and return the result.
   * Return `nullptr` if dimensions mismatch for input matrices.
   * @param matrixA Input matrix
   * @param matrixB Input matrix
   * @return The result of matrix multiplication
   */
  static std::unique_ptr<RowMatrix<T>> Multiply(const RowMatrix<T> *matrixA, const RowMatrix<T> *matrixB) {
    // TODO(P0): Add implementation
    if (matrixA->GetColumnCount() != matrixB->GetRowCount()) {
      return std::unique_ptr<RowMatrix<T>>(nullptr);
    }
    int row = matrixA->GetRowCount();
    int col = matrixB->GetColumnCount();
    int k_size = matrixA->GetColumnCount();
    auto matrix_c = std::make_unique<RowMatrix<T>>(row, col);
    T sum;
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        sum = 0;
        for (int k = 0; k < k_size; k++) {
          sum += matrixA->GetElement(i, k) * matrixB->GetElement(k, j);
        }
        matrix_c->SetElement(i, j, sum);
      }
    }
    return matrix_c;
  }

  /**
   * Simplified General Matrix Multiply operation. Compute (`matrixA` * `matrixB` + `matrix_c`).
   * Return `nullptr` if dimensions mismatch for input matrices.
   * @param matrixA Input matrix
   * @param matrixB Input matrix
   * @param matrix_c Input matrix
   * @return The result of general matrix multiply
   */
  static std::unique_ptr<RowMatrix<T>> GEMM(const RowMatrix<T> *matrixA, const RowMatrix<T> *matrixB,
                                            const RowMatrix<T> *matrix_c) {
    // TODO(P0): Add implementation
    auto matrix_d = Multiply(matrixA, matrixB);
    if (matrix_d != nullptr) {
      return Add(matrix_c, matrix_d);
    }
    return std::unique_ptr<RowMatrix<T>>(nullptr);
  }
};
}  // namespace bustub
