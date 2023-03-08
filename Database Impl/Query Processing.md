- Happen after parsing the query + internalizing the data
- Query processor: turns user queries and data modification commands into a query plan
	Decisions made by the processor:
	- Which of the algebraically equivalent forms of a query is the most efficient algorithm?
	- For each algebraic operator, what algorithm should we use?
	- How should the operators pass data from one to the other? (e.g. main memory buffers, disk buffers)
- ### How to execute the query?
>	SELECT B,D
>	FROM R,S
>	WHERE R.A = "C" AND S.E = 2 AND R.C=S.C
1. Compute Cartesian product "$\times$": Compute every possible ways, and scan
		![[Pasted image 20230307161117.png]]
		- On the fly: Not buffering into the memory buffer
		- The Cartesian product is too expensive in memory
2. Natural Join
		![[Pasted image 20230307161644.png]] ![[Pasted image 20230307162343.png]]
3. Right Index Natural Join
		![[Pasted image 20230307162511.png]]
- ### Issues in Query Processing and Optimization
	- Generate Plans
		Emply efficient execution primtives for computing relational algebra ops
		Systematcially transform expressions to achieve more efficient combinations of ops
	 - Estimate Cost of Generated Plans
		 Statistics
	 - "Smart" Search of the Space of Possible Plans
		 Always do the "good" transformations (relational algebra optimization)
		 Prune the space (e.g., System R)
	- Often the above steps are mixed
- ### The Jouney of a Query
	![[Pasted image 20230307164208.png]]
	- #### Logical query plan generator
		- Old-fassion one-to-one conversion:
		![[Pasted image 20230307165118.png]]
		- With transformation:
		![[Pasted image 20230307165326.png]] ![[Pasted image 20230307165715.png]]
	- #### Physical query plan generator:
		![[Pasted image 20230307170058.png]]
		- INDEX(input constant) := index of the tuple/entry
		- LEFT INDEX(right table input constant) := index of the tuple/entry with the same variable value in the left table
		![[Pasted image 20230307170248.png]]
		- ##### Summary
			- In general, more than one different logical plans may be generated by choosing different primitives
			1. Usually, we generate a plan based on some heuristics
			2. We then estimate its timecost
			3. Use small portion of that time to search for better alternatives

e.g.
- The query:
 SELECT title
 FROM Starsln
 WHERE starName IN (
	 SELECT name
	 FROM MovieStar
	 WHERE birthdate LIKE '%1960')
- The parser
	![[Pasted image 20230307170940.png]]
- Algebraic plan
	![[Pasted image 20230307170959.png]]
- Logical plan and improvements
	![[Pasted image 20230307171026.png]]
	![[Pasted image 20230307171047.png]]
- Size estimation
	Estimate the size of each op's results
- Physical plan
	![[Pasted image 20230307171143.png]]
- 