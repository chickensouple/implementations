#include <vector>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <iostream>
#include <stdexcept>

// Min Binary Heap with HashMap for fast update_priority() and remove() operations

template <class T, class P, class Comp = std::less<P>>
class HashHeap {
public:
    HashHeap();
    ~HashHeap();

    void push(const T& obj, const P& priority);
    T pop();
    size_t size() const;
    bool empty() const;
    bool update_priority(const T& obj, const P& new_priority);
    bool remove(const T& obj);

    void debug_print() const;

private:
    std::vector<std::pair<T, P>> _heap;
    std::unordered_map<T, size_t> _map;
    Comp _comp;

    void _swap(size_t idx1, size_t idx2);

    /**
     * @brief moves the object at an index up the tree (towards the begining of the heap)
     * until it is less than its children
     * 
     * @param idx index of object to sift up
     */
    void _sift_up(size_t idx);

    /**
     * @brief moves the object at an index down the tree (towards the end of the heap)
     * until it is greater than or equal to its parent
     * 
     * @param idx index of object to sift down
     */
    void _sift_down(size_t idx);
};


template <class T, class P, class Comp>
HashHeap<T, P, Comp>::HashHeap() : _comp(Comp()) {}

template <class T, class P, class Comp>
HashHeap<T, P, Comp>::~HashHeap() {}

template <class T, class P, class Comp>
void HashHeap<T, P, Comp>::push(const T& obj, const P& priority) {
    _heap.push_back(std::make_pair(obj, priority));
    _map[obj] = _heap.size() - 1;

    _sift_up(_heap.size() - 1);
}

template <class T, class P, class Comp>
T HashHeap<T, P, Comp>::pop() {
    auto pair = _heap[0];
    _swap(0, _heap.size()-1);

    _heap.pop_back();
    _map.erase(pair.first);
    
    _sift_down(0);
    return pair.first;
}

template <class T, class P, class Comp>
size_t HashHeap<T, P, Comp>::size() const {
    return _heap.size();
}


template <class T, class P, class Comp>
bool HashHeap<T, P, Comp>::empty() const {
    return _heap.empty();
}

template <class T, class P, class Comp>
bool HashHeap<T, P, Comp>::update_priority(const T& obj, const P& new_priority) {
    auto search = _map.find(obj);
    if (search == _map.end()) {
        throw std::runtime_error("No such key exists!");
    }

    size_t idx = search->second;
    P old_priority = _heap[idx].second;
    _heap[idx] = std::make_pair(obj, new_priority);
    if (_comp(new_priority, old_priority)) {
        // if new priority is less than old one, sift up
        _sift_up(idx);
    } else {
        // otherwise, sift down
        _sift_down(idx);
    }
    return true;
}

template <class T, class P, class Comp>
bool HashHeap<T, P, Comp>::remove(const T& obj) {
    auto search = _map.find(obj);
    if (search == _map.end()) {
        throw std::runtime_error("No such key exists!");
    }

    size_t idx = search->second;
    P old_priority = _heap[idx].second;

    // remove object from heap and map
    // and move last element into the spot
    _swap(idx, _heap.size()-1);
    _heap.pop_back();
    _map.erase(obj);

    // sift the the new element around
    P new_priority = _heap[idx].second;
    if (_comp(new_priority, old_priority)) {
        // if new priority is less than old one, sift up
        _sift_up(idx);
    } else {
        // otherwise, sift down
        _sift_down(idx);
    }
    return true;
}

template <class T, class P, class Comp>
void HashHeap<T, P, Comp>::debug_print() const {
    std::cout << "Heap\n";
    std::cout << "===============\n";
    std::cout << "(Obj, Priority)\n";
    for (size_t i = 0; i < _heap.size(); i++) {
        std::cout << "(" << _heap[i].first << ", " << _heap[i].second << ")\n";
    }
    std::cout << "\nMap\n";
    std::cout << "===============\n";
    for (auto i = _map.begin(); i != _map.end(); i++) {
        std::cout << i->first << ": " << i->second << "\n";
    }

}

template <class T, class P, class Comp>
void HashHeap<T, P, Comp>::_swap(size_t idx1, size_t idx2) {
    _map[_heap[idx1].first] = idx2;
    _map[_heap[idx2].first] = idx1;
    std::swap(_heap[idx1], _heap[idx2]);
}

template <class T, class P, class Comp>
void HashHeap<T, P, Comp>::_sift_up(size_t idx) {
    // swap elements until root is reached or until it is greater or equal than parent
    size_t i = idx;
    while (i != 0) {
        size_t parent_i = (i-1) / 2;

        if (_comp(_heap[i].second, _heap[parent_i].second)) {
            _swap(i, parent_i);
            i = parent_i;
        } else {
            break;
        }
    }
}

template <class T, class P, class Comp>
void HashHeap<T, P, Comp>::_sift_down(size_t idx) {
    size_t i = idx;
    
    // swap elements until there are no more children in range
    // or until it is less than all children
    while (2*i + 1 < _heap.size()) {
        size_t child1_i = 2*i + 1;
        size_t child2_i = 2*i + 2;

        size_t min_child_i; // index to child with smallest priority

        if (child2_i >= _heap.size()) {
            // if there is no second child
            min_child_i = child1_i;
        } else {
            // find minimum of two children
            min_child_i = _comp(_heap[child1_i].second, _heap[child2_i].second) ? child1_i : child2_i;
        }
  
        if (_comp(_heap[i].second, _heap[min_child_i].second)) {
            break;
        } else {
            _swap(i, min_child_i);
            i = min_child_i;
        }
    }
}

