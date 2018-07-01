#include <iostream>
#include <string>
#include "hashheap/hashheap.hpp"

struct Person {
    std::string name;
    int age;

    Person(std::string name_in, int age_in) : name(name_in), age(age_in) {};
};



struct TestHash {
    TestHash() {};

    size_t operator()(const Person* p) const {return size_t(p);};
};


// struct TestHashInt {
//     size_t operator()(const int& i) const {return i;};
// };

// struct TestHashInt2 {
//     size_t operator()(const int* i) const {return (size_t)i;};
// };


int main(void) {


    // auto hashheap = HashHeap<int*, int, std::less<int>, TestHashInt2>();
    // hashheap.push(new int(2), 1);
    // hashheap.push(new int(3), 0);


    // // norm, greg, bob, alice, lydia, richard
    // while (not hashheap.empty()) {
    //     int* obj = hashheap.pop();
    //     std::cout << *obj << "\n";
    // }



    std::cout << "hello world!\n";

    Person* bob = new Person("bob", 4);
    Person* alice = new Person("alice", 6);
    Person* richard = new Person("richard", 7);
    Person* greg = new Person("greg", 87);
    Person* norm = new Person("norm", 9);
    Person* lydia = new Person("lydia", 81);
    Person* sarah = new Person("sarah", 21);

    auto hashheap = HashHeap<Person*, int, std::less<int>, TestHash>();
    hashheap.push(bob, 1);
    hashheap.push(alice, 4);
    hashheap.push(richard, -1);
    hashheap.push(greg, 0);
    hashheap.push(norm, 2);
    hashheap.push(lydia, 5);
    hashheap.push(sarah, 3);

    hashheap.update_priority(norm, -2);
    hashheap.update_priority(richard, 6);
    hashheap.remove(sarah);

    // norm, greg, bob, alice, lydia, richard
    while (not hashheap.empty()) {
        Person* obj = hashheap.pop();
        std::cout << obj->name << ": " << obj->age << "\n";
    }
}


