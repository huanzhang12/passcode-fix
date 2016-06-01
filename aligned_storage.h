/*
 * All Copyright Reserved 
 *
 * Author: Hsiang-Fu Yu (rofuyu@cs.utexas.edu)
 *
 * */
#ifndef ALIGNED_STORAGE_H
#define ALIGNED_STORAGE_H

#include <vector>
#include <cstdlib>
#include <cstddef>

#define CACHE_LINE_SIZE 64

template <class T, int align_size=CACHE_LINE_SIZE>
struct alignize { // {{{
	//T val alignas(align_size);
	T val __attribute__((aligned(align_size)));
	alignize() {}
	alignize(const T& val_): val(val_) {}
	alignize(const alignize& b): val(b.val) {}
	alignize(alignize& b): val(b.val) {}
}; // }}}

template<typename T, int align_size=CACHE_LINE_SIZE>
class memalign_allocator { // {{{
	public:
		typedef T value_type;
		typedef value_type* pointer;
		typedef const value_type* const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		template<typename U> struct rebind { typedef memalign_allocator<U> other; };

		memalign_allocator() throw() {}
		memalign_allocator( const memalign_allocator& ) throw() {}
		template<typename U> memalign_allocator(const memalign_allocator<U>&) throw() {}

		pointer address(reference x) const {return &x;}
		const_pointer address(const_reference x) const {return &x;}

		pointer allocate( size_type n, const void* /*hint*/=0 ) {
			pointer alloc_ptr;
			int ok = posix_memalign((void**)&alloc_ptr, align_size, n*sizeof(value_type));
			if(ok != 0) {
				printf("align_size %d val %ld\n", align_size, n * sizeof(value_type));
				throw std::bad_alloc();
			}
			return alloc_ptr;
		}
		void deallocate( pointer p, size_type ) { free(p); }

		//! Largest value for which method allocate might succeed.
		size_type max_size() const throw() { return (~size_t(0)-align_size)/sizeof(value_type); }

		template<typename U, typename... Args>
		void construct(U *p, Args&&... args) {::new((void *)p) U(std::forward<Args>(args)...);}
		//void construct(U *p, Args&&... args) {::new((void *)p) U((args)...);}
		void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
		void destroy( pointer p ) {p->~value_type();}
};

template<> 
class memalign_allocator<void> {
	public:
		typedef void* pointer;
		typedef const void* const_pointer;
		typedef void value_type;
		template<typename U> struct rebind {
			typedef memalign_allocator<U> other;
		};
}; 

template<typename T, typename U>
inline bool operator==( const memalign_allocator<T>&, const memalign_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const memalign_allocator<T>&, const memalign_allocator<U>& ) {return false;}
// }}}

template <class T, int align_size=CACHE_LINE_SIZE, class Alloc=memalign_allocator<T, align_size> >
class aligned_vector: public std::vector<alignize<T>, Alloc> { // {{{
		typedef std::vector<alignize<T>, Alloc> super;
		class iterator : public super::iterator{
			protected:
				alignize<T>* ptr;
			public:
				iterator(alignize<T>* ptr_): ptr(ptr_){}
				T& operator*() const { return ptr->val; }
				T* operator->() const { return &(ptr->val); }
				iterator& operator++() {++ptr; return *this; }
				iterator& operator++(int) { return iterator(ptr++); }
				bool operator!=(const iterator& other) const { return ptr!=other.ptr; }
		};
	public:
		aligned_vector(size_t size=0){ super::resize(size, T()); }
		aligned_vector(size_t size, const T& val){ super::resize(size, val); }
		T& operator[](size_t idx) { return super::operator[](idx).val; }
		T& at(size_t idx) { return super::at(idx).val; }
		const T& operator[](size_t idx) const { return super::operator[](idx).val; }
		const T& at(size_t idx) const { return super::at(idx).val; }
		void resize(size_t size) { super::resize(size, alignize<T>(T())); }
		void resize(size_t size, const T& val) { super::resize(size, alignize<T>(val)); }
		void push_back(const T& val) { super::push_back(alignize<T>(val)); }
		iterator begin() { return iterator(&super::at(0)); }
		iterator end() { return iterator(&super::at(0)+super::size()); }
		void reset(const T& val) { for(auto &x: *this) x = val; }
		T sum(){
			size_t len = super::size();
			if(len == 0) return T();
			T ret = at(0);
			for(size_t i = 1; i < len; ++i) ret += at(i);
			return ret;
		}
		T max(){
			size_t len = super::size();
			if(len == 0) return T();
			T ret = at(0);
			for(size_t i = 1; i < len; i++) 
				if(at(i) > ret) ret = at(i);
			return ret;
		}
		T min(){
			size_t len = super::size();
			if(len == 0) return T();
			T ret = at(0);
			for(size_t i = 1; i < len; i++)
				if(at(i) < ret) ret = at(i);
			return ret;
		}
}; // }}}

#endif // ALIGNED_STORAGE_H
