When does index help? Consider the selection query of a table with $n$ tuples and $m$ distinct R.A attribute
```
SELECT \*
FROM R
WHERE R.A = ?
```
Without index the operation is $\in O(n)$ since we go through and check each tuples.
An index on R.A is a data structure that answers very efficiently the request "find the tuples with R.A = $c$". Then a query is answered $\in O(k)$ where $k$ is the number of typles with R.A = $c$. Therefore the *expected* time cost is $O(n/m)$

### How to create/drop an index on R.A?
```
CREATE INDEX myIndex ON R(A)
DROP INDEX myIndex
```
One do not need to change queries to invoke/ignore using indices -> database auto-handled

### Indexing
Data Structures used for quickly locating tuples by conditions
	- Equality condition
	- Other conditions e.g. range
Many types of indexes. Evaluate them on
	- Acces time
	- Insertion time
	- Deletion time
	- Space needed
Should I build an idex? Has to take maintenance (e.g. index update) cost into account!
	In OLAP it seems beneficial to create an index on R.A whenever $m>1$
	Some suprisingly efficient algorithms that do not use indices
Non-volatile storage is important to OLTP even when RAM is large
- Persistence important for transaction atomicity and durability
- Even if database fits in main memory changes have to be written in non-volatile storage
- Hard disk
- RAM disks w/ battery
- Flash memory
Peculiarities of storage mediums affect algorithm choice
- **Block-based access: (what we focus on)**
	Access performance: How many blocks were accessed
- Clustering for sequential access:
	Accessing consecutive blocks costs less on disk-based system
- Recall Moore's Law:
	$$\frac{\text{Disk/RAM Transfer Rate}}{\text{Disk/RAM Access Speed}} \text{\ grows exponentially}$$
- Example
	![[Pasted image 20230308111628.png]]
	Our objective (time cost) now also include minimizing the number of time accessing blocks
	![[Pasted image 20230308111827.png]]
	Sort each consecutive blocks that fits in RAM, and store in 2rd storage
	![[Pasted image 20230308111949.png]]
	And merge the keys 
	![[Pasted image 20230308112340.png]]
	- Most files can be sorted in only 2 passes!
		Assume
		- $M$ bytes of RAM buffer
		- $B$ bytes per block
		Calculation:
		- The assumption of Phase 2 holds when \#file < $M/B$
			There can be up to $M/B$ Phase 1 rounds
		- Each round can process up to $M$ bytes of input data
			2 Phase Merge Sort can sort $M^2/B$ bytes
			E.g. $(8GB)^2/64KB=1TB$
### Horizontal placement of SQL data in blocks
Relations:
- Pack as many tuples per block - improves scan time
- Do not reclaim deleted records
- Utilize **overflow records** if relation must be sorted on promary key
- A novel generation of databases feature column storage!
![[Pasted image 20230308120150.png]]
![[Pasted image 20230308120136.png]]
### Secondary storage
- Conventional indices
	As a thought experiment (tho ideas are integrated to industries)
- B-trees
	The workhorse of most db systems
- **Hashing schemes: easiest to implement, strongly recommended for Milestone3**
- Bitmaps
	An analytics favorite

### Conventional indices
### Terms
- Primary index
	The index on the attribute (aka search key) that determines the sequencing of the table
- Secondary index
	Index xon any other attribute
- Dense index
	Every value of the indexed attribute appears in the index
- Sparse index
	Many values do not appear
- Examples 
	![[Pasted image 20230308120945.png]]
- Multilayer index
	![[Pasted image 20230308125218.png]]
- Representation of duplicate values in primary indexes
	Index may point to first instance of each value only
	![[Pasted image 20230308125427.png]]

- ### Maintainance of index
	- #### Deletion 
		- Dense primary index file with no duplicate values is handled in the same way with from a sequential file
			![[Pasted image 20230308125651.png]]
		- Sparse index
			- If the deleted entry does not appear in the index do nothing
			![[Pasted image 20230308130215.png]]
			- Else replace it with the next search-key value
			![[Pasted image 20230308130229.png]]
			- Unless the next search key value has its own index entry. In this case delete the entry
			![[Pasted image 20230308130107.png]]
	- #### Insertion
		- Sparse:
			- If no new block is created then do nothing
			- Else create overflow record
				- Reorganize periodically
				- Could we claim space of next block?
				- How often do we reorganize and how much expensive it is?
				- B-trees offer convincing answers
			![[Pasted image 20230308130946.png]]

### Unsorted Secondary Indexes with Duplicate Values
If not sorted, cannot have a sparse index on top of that, bucause we assume it is sorted.
![[Pasted image 20230308131144.png]]
- Option 1: linked pointers to fields - not work for ranging condition
- Option 2: linked nodes in secondary keys to the next key - additional fields to store & need to follow chain
- Option 3: an intermediate buckets to build associativity
![[Pasted image 20230308132509.png]]
	Why "bucket"?
		It enables the processing of queries working with pointers only
		![[Pasted image 20230308132815.png]]
		Very common technique in Info Retrieval

### Summary of Indexing So Far
- Basic topics in conventional indexes
	- Multiple levels
	- Sparse/dense
	- Duplicate keys and buckets
	- Deletion/insertion similar to sequential files
- Advantages
	- Simple algorithms
	- Index is sequential file
- Disadvantages
	- Eventually sequantiality is lost due to overflows, reorganizations are needed but costy
		![[Pasted image 20230308133812.png]]

### B+tree
Example
	![[Pasted image 20230308135600.png]]
	![[Pasted image 20230308140038.png]]
	![[Pasted image 20230308140045.png]]
- ##### Size of nodes:
	- $n+1$ pointers and $n$ keys
- ##### Non-root nodes have to bee at least half-full
	Use at least
	- Non-leaf: $\lceil(n+1)/2\rceil$ pointers to nodes
	- Leaf: $\lfloor(n+1)/2\rfloor$ pointers to data
- ##### Rules of tree of order $n$:
	1. Balanced tree - All leaves at same lowest level
	2. Pointers in leaves pointto records except for 'sequence pointer'
	3. Number of pointers/keys for B+tree
		![[Pasted image 20230308141018.png]]
- ##### Insert into B+tree
	1. Insert key
	2. 