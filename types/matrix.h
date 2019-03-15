/*
    This file defines the Matrix class. Matrix supports creation of n dimensional arrays
    It supports various arithmetic and matrix operations.
*/
#include "utils/common.h"
#include "overloads/overloads.h"

#ifndef __MATRIX_FLOAT_INCLUDED__   
#define __MATRIX_FLOAT_INCLUDED__  

template<typename T>
struct Matrix{
    private:
    
    /* 
        This vector tracks for each dimension- the total number of elements 
        left to encounter if I continue deeper into the remaining dimensions.
    */
    vector<int> elemsEncounteredPerDim;

    // Verifies that the shape provided and val vector provided are compatible in size
    bool verifyShape(const vector<T> &val, const vector<int> &shape) {
        bool s = true;
        int p = 1;
        for(auto i: shape) {
            p = p*i;
        }
        if(p != val.size()) {
            s = false;
        }

        return s;
    }
    
    /* 
        This function is used to compute the elemsEncounteredPerDim variable. 
        See it's description for more detail
    */
    vector<int> computeShapes(const vector<int> &shapes) {
        vector<int> elems(shapes.size());
        int p = 1;

        for(int i = shapes.size()-1; i >=0;i--) {
            elems[i] = p;
            p *= shapes.at(i);
        }

        return elems;
    }

    /*
        This function checks that the 2 matrices are compatible with each other to perform elementwise 
        operations. Operations such as *, + and / are elementwise operations and they require the two 
        matrices to have the same shape. 
    */
    bool verifyShapeForElementwiseOperation(const vector<int> &shape1, const vector<int> &shape2) {
        
        int n = shape1.size();
        int m = shape2.size();

        if(n != m) {
            return false;
        }

        for(int i = 0;i<n;i++) {
            if(shape1.at(i) != shape2.at(i)) {
                return false;
            }
        }

        return true;
    }

    /*
        This function verifies that the 2 matrices are compatible for dot product multiplication.
        Currently, this dot product functionality allows the lhs matrix (with shape1) to have 2 or 
        more dimensions. The rhs matrix should have 2 dimensions. The multiplication will occur by 
        traversing through the n-2 dimensions of matrix 1, until we get a 2d matrix of lhs. Then we
        perform matrix multiplication with this 2 d lhs matrix and the rhs matrix. This function
        performs a check to verify whether the two matrices are compatible for matrix multiplication.
    */
    bool verifyShapeForDotProductOperation(const vector<int> &shape1, const vector<int> &shape2) {
        int n = shape1.size();
        int m = shape2.size();

        if(n < 2 || m != 2) {
            return false;
        } 

        auto col1 = shape1.at(n-1);
        auto row1 = shape1.at(n-2);
        auto col2 = shape2.at(m-1);
        auto row2 = shape2.at(m-2);

        if(col1 != row2) {
            return false;
        }

        return true;
    }

    /*
        This is a utility function for matrix multiplication that performs the actual dot product.
        It gets as input the reference of the result vector, the 2d rhs matrix, the start index that 
        tracks the position from which to start in the n dimensional lhs matrix and resStart tracks
        the position to start in the result vector 
    */
    void matmulUtil(vector<T> &res,
            const Matrix<T> &rhs,
            int start, int startRes) {
        int row1 = this->shape[this->shape.size()-2];
        int col1 = this->shape[this->shape.size()-1];
        int row2 = rhs.shape[rhs.shape.size()-2];
        int col2 = rhs.shape[rhs.shape.size()-1];
        // Sanity Check
        assert(col1 == row2);

        // O(n^3) complexity 
        for(int i = 0;i<row1;i++) {
            for(int k = 0;k < col2;k++) {
                T sum = 0;
                for(int j = 0;j< col1;j++) {
                    sum += this->val[start+ i*col1 + j] * rhs.val[j*col2 + k];
                }
                res[(startRes+ i*col2 + k)] = sum;
            }
        }
    }

    /*
        This function performs the matrix multiplication between a n dimensional matrix and a 2d matrix.
        To view it's usage check the dot() function from where it is called. It is a recursive function
        that calls itself until two 2d matrices have been got. It keeps a track of the position of nd 
        matrix by using a stack. The stack keeps a track of the index of each dimension that has been
        traversed. We use this information to find the index from where to start the 2d matrix 
        multiplication. To find out the index from where to do the  2D matrix multiplication, we shall 
        look into an example:

        Assume we have 2 matrices of shape A = (3,4,2,1) and B = (1,2)
        Assume we are currently at position [2][3] in our lhs matrix.
        C = A[2][3];

        Hence we are ready to do our 2d matrix multiplication as we have two 2d matrices C and B now.
        But since our matrix is stored as a 1d vector , we need to find the index from where to start
        the multiplicatoinn for A.
        The right index would be 2*(num_col_for_dimension_0) + 3*(num_col_for_dimension_1).

        These num cols for each dimension are stored in the elemsEncounteredPerDim vector.
        Hence we use this information to find out the correct starting index of A.

        We use a similiar thing to track the starting index of result vector by computing 
        num_cols_per_dimension in resElems.

        Lastly the dim parameter tracks what the current dimension is. IT start from 0 and keeps 
        on being incremented after each function call.

    */
    void matmul(vector<T> &res,
                    const Matrix<T> &rhs,
                    vector<int> &stack,
                    const vector<int> &resElems,
                    int dim) {
        // Case when both matrices are 2D
        if(stack.size() == shape.size()-2) {
            int p = 0;
            int s = 0;
            for(int i = 0;i < stack.size();i++) {
                p += elemsEncounteredPerDim[i]*stack[i];
                s += resElems[i]*stack[i];
            }
            // Do normal 2D matrix multiplication now
            matmulUtil(res,rhs,p,s);
            return;
        }
        // Case when larger matrix is bigger than rhs matrix
        for(int i = 0;i < this->shape.at(dim);i++) {
            stack.push_back(i); // Calculates how many elements have been processed, to get pointer to right location in val and pushes to stack
            matmul(res,rhs,stack,resElems,dim+1);
            stack.pop_back(); // Pops out of stack
        }
    }

    public:
    vector<T> val; // underlying data structure that holds the values of matrix
    vector<int> shape; // shape of that matrix
    
    /* 
        Need this default constructor without which program seems to error out
        TODO: needs investigatio on why this happening
    */
    Matrix() {
        //Default
    }

    // Constructor for matrix, validates shape before creating object
    Matrix(vector<T> val, vector<int> shape) {
        // Deep Copy
        assert("Shape and size of vector are incompatible !" && verifyShape(val,shape));
        this->val = val;
        this->shape = shape;
        this->elemsEncounteredPerDim = computeShapes(shape);
    }

    /* 
        Print's out Matrix in human readable format, useful debugging small matrices.
        This function is used by the overloaded << operator. Foloows a simialr structure to the matmul
        function
    */
    ostream & print(ostream &out, vector<int> &stack, int dim) {
        if(stack.size() == shape.size()-1) {
            int p = 0;
            for(auto i = 0;i < stack.size();i++) {
                p += stack[i];
            }
            out<<"[ ";
            for(auto i=0; i< this->shape.at(dim); i++) {
                out<<this->val.at(p+i)<<" ";
            }
            out<<"]";
            return out;
        }

        out<<"[ ";
        for(auto i = 0;i < this->shape.at(dim);i++) {
            stack.push_back(elemsEncounteredPerDim[dim]*i); // Calculates how many elements have been processed, to get pointer to right location in val and pushes to stack
            print(out,stack,dim+1);
            stack.pop_back(); // Pops out of stack
        }
        out<<" ]";
        return out;
    }

    // Performs elementwise addition
    Matrix<T> operator + (const Matrix<T> &rhs) {
        assert("Shapes aren't compatible for addition !" &&
         verifyShapeForElementwiseOperation(this->shape, rhs.shape));

        auto res = this->val + rhs.val;
        auto resShape = this->shape;
        return Matrix(res, resShape);
    }

    // Performs elementwise multiplication
    Matrix<T> operator * (const Matrix<T> &rhs) {
        assert("Shapes aren't compatible for multiplication !" &&
         verifyShapeForElementwiseOperation(this->shape, rhs.shape));

        auto res = this->val * rhs.val;
        auto resShape = this->shape;
        return Matrix(res, resShape);
    }

    // Performs elementwise division
    Matrix<T> operator / (const Matrix<T> &rhs) {
        assert("Shapes aren't compatible for division !" && 
        verifyShapeForElementwiseOperation(this->shape, rhs.shape));

        auto res = this->val / rhs.val;
        auto resShape = this->shape;
        return Matrix(res, resShape);
    }

    // Performs power operation with scalar
    Matrix<T> operator ^ (const T &rhs) {
        auto res = this->val ^ rhs;
        auto resShape = this->shape;
        return Matrix(res, resShape);
    }

    // Perfroms exponent operation on each value of matrix
    Matrix<T> exp() {
        auto res = exponent(this->val);
        auto resShape = this->shape;
        return Matrix(res, resShape);
    }

    // Dot Product between 2 matrices. Check matmul for indepth description.
    Matrix<T> dot(const Matrix<T> &rhs) {
        assert("Shapes aren't compatible for dot product !" && 
        verifyShapeForDotProductOperation(this->shape, rhs.shape));
        
        // Calc size of res array
        auto size = 1; // total size of resultant matrix
        vector<int> resShape;
        for(auto i = 0;i < this->shape.size()-1;i++) {
            size *= this->shape[i];
            resShape.push_back(this->shape[i]);
        }

        size*=rhs.shape[rhs.shape.size()-1];
        resShape.push_back(rhs.shape[rhs.shape.size()-1]);

        vector<T> res(size,0);
        vector<int> stack;

        auto resElems = computeShapes(resShape);

        matmul(res,rhs,stack,resElems,0);

        return Matrix(res, resShape);
    }


    // Delete matrix
    ~Matrix() {
        
    }
};

// Overloaded function for cout<<matrix<<endl;
template<typename T>
ostream & operator << (ostream &out, Matrix<T> &m) {
    vector<int> stack;
    return m.print(out,stack,0);
}

// Divison with a scalar as divident
template<typename E>
Matrix<E> operator / (const E e, const Matrix<E> &rhs) {
    auto res =  e/rhs.val;
    auto resShape = rhs.shape;
    return Matrix<E>(res, resShape);
}

#endif

