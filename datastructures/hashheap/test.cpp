#include <iostream>
#include <string>
#include "hashheap.hpp"

struct Person {
    std::string name;
    int age;

    Person(std::string name_in, int age_in) : name(name_in), age(age_in) {};
};

int main(void) {

    std::cout << "hello world!\n";

    Person* bob = new Person("bob", 4);
    Person* alice = new Person("alice", 6);
    Person* richard = new Person("richard", 7);
    Person* greg = new Person("greg", 87);
    Person* norm = new Person("norm", 9);
    Person* lydia = new Person("lydia", 81);
    Person* sarah = new Person("sarah", 21);

    auto hashheap = HashHeap<Person*, int>();
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

 
