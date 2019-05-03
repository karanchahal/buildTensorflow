/*
    This file defines the Stochastic Gradient Descent Optimiser. The Stochastic Gradient Descent
    Optimizer takes the loss computed over a single training example or the averages of the loss
    computed with multiple training examples and "minimises" the loss.

    By minimising, we mean it finds out all the updatable tensors that contributed towards 
    computing this loss. Once it has these parameters it performs an update step on each 
    parameter (Tensor) to tweak them into the right direction to minimise the overall loss.

    It performs this update step by this formula:
    
    val = val - learning_rate*gradient_of_val
    
    Where val is the value of the tensor and gradient_of_val is the partial gradient of the
    tensor with respect to the loss.
*/

#include "optims/optim.h"
#include <queue>

#ifndef __OPTIM_SGD_INCLUDED__   
#define __OPTIM_SGD_INCLUDED__ 


template<typename T>
class SGD : public Optimizer<T> {

    public: 

    bool debug = true;

    SGD(T lr, bool debug=true) {
        this->params.clear();
        this->lr = lr;
        this->debug = debug;
    }

    /*
        This function does a full search through the computational graph of the Tensor x and
        stores all the Tensor nodes of the graph in the params set.

        The params set represents all the tensors that need t be updated.

        As of now, a BFS style algorithm traverses through the graph to find out all the Tensor
        nodes.
    */
    void getParams(Tensor<T>* x) {
        
        this->params.clear(); // Clear out old params. Should we do this ? 
        
        queue<Tensor<T>*> q;
        q.push(x);

        while(!q.empty()) {

            auto v = q.front();
            q.pop();
            auto op = v->backOp;

            if(op) {

                if(op->t1 != NULL && this->params.find(op->t1) == this->params.end() &&
                    op->t1->requires_grad) {
                    q.push(op->t1);
                    this->params.insert(op->t1);
                }

                if(op->t2 != NULL && this->params.find(op->t2) == this->params.end()
                    && op->t2->requires_grad) {
                    q.push(op->t2);
                    this->params.insert(op->t2);
                }
            }
        }
    }

    /*
        This function is the function all users will use to perfrom the gradient descent update
        for their model. It performs this operation in 3 phases.
        1. Gets all tensor parameters
        2. Updates all these parameters via the step function
        3. Clear's all the gradients of the parameters for the next step.
    */
    void minimise(Tensor<T>* x) {

        // Get all tensors in computational graph
        getParams(x);

        // step through 1 parameter update
        step(this->lr);

        // reset Gradients to zero
        this->zeroGrad();
       
    }

    void plot(Tensor<T> *t) {
        int total = t->grad.val.size();
        float zeros = 0;
        for(auto i: t->grad.val) {
            if (i == 0) {
                zeros++;
            }
        }

        auto percentZeros = zeros*10/total;
        
        cout<<t->name<<" | ";
        for(int i = 0;i <10;i++) {
            if(i < percentZeros) {
                cout<<"#";
            } else {
                cout<<" ";
            }
        }
        cout<<" |"<<endl;

    }

    // Performs 1 step of gradient descent. See top of the file to see definition of SGD. 
    void step(T learning_rate) {

        if(debug) {   
            // cout<<"Number of parameters updated: "<< this->params.size()<<endl;
            cout<<"----------------------------------------------------"<<endl;
        }
        // cout<<"Names of these parameters are: "<<endl;
        for(auto t: this->params) {
            // cout<<t->name<<endl;
            if(debug) {
                plot(t);
            }
            t->val = t->val - learning_rate*t->grad;
        }
        
        if(debug) {
            cout<<"----------------------------------------------------"<<endl;
        }


    }

};

#endif
