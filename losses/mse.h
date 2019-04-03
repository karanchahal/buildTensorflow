
template<typename T>
Tensor<T> * mse_loss(Tensor<T>* y, Tensor<T>* gt) {
    return sqrt(y - gt);
}


template<typename T>
Tensor<T> * huber_loss(Tensor<T>* y, Tensor<T>* gt) {
    // TODO
}


template<typename T>
Tensor<T> * log_loss(Tensor<T>* y, Tensor<T>* gt) {
    // TODO
}

template<typename T>
Tensor<T> * cross_entropy_loss(Tensor<T>* y, Tensor<T>* gt) {
    // TODO
}


template<typename T>
Tensor<T> * nll_loss(Tensor<T>* y, Tensor<T>* gt) {
    // TODO
}


template<typename T>
Tensor<T> * hinge_loss(Tensor<T>* y, Tensor<T>* gt) {
    // TODO
}

