# 15618-final-project
Proposal link: https://drive.google.com/file/d/1H6Dh6pzju2yUPYrLQwE_7uuU_qGkSM_r/view?usp=sharing
### SUMMARY
We are going to implement paralleled Fast Fourier transform algorithm on both GPU and multi-core
CPU platforms, and perform a detailed analysis of both systems’ performance characteristics.
### BACKGROUND
A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT)
of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain
(often time or space) to a representation in the frequency domain and vice versa. [1]
Fast Fourier transforms are widely used for applications in engineering, music, science, and mathe-
matics. [1] One example is big integer multiplication: with FFT, the time complexity of big integer
multiplication can be reduced from O(n2) to O(nlogn).
Moreover, FFT can be parallelized, which can help accelerate big integer multiplication even more.
### CHALLENGE
1. FFT involves multiple iterations, which may cause synchronization over head when imple-
menting parallel version.
2. The computation requires information from different locations, which brings trouble when
trying to utilizing locality, especially when running on GPU.
3. Divergent executions happens all the time.
### RESOURCES
We are not using starter codes for our implementation of Fast Fourier transform algorithm on CPU,
but might use relevant codes for our implementation on GPU (several papers regarding this topic),
and might use relevant codes for Fast Fourier transform algorithm application if we achieves our
125 percent target.
We are using MAC computers and GHC/PSC computers for CPU/GPU implementations.
### GOALS AND DELIVERABLES
75 percent: Implement parallel FFT on CPU to achieve reasonable speed-ups, explore possibility of efficient caching and even workload distribution, analyze the performance. 
100 percent: Implement parallel FFT on CPU to achieve reasonable speed-ups, explore possibility of efficient caching and even workload distribution, analyze the performance. Implement parallel FFT on GPU and analyze the performance, determine the bottleneck.
125 percent: Achieve reasonable speed-ups for GPU implementation of FFT, and explore its application (on image/video processing).
Delivery: We will show speed-up graphs from our analysis.
### PLATFORM CHOICE
We are implementing our code in C++. MAC computer are for small to medium level CPU testing,
GHC machine are good enough for small to medium level GPU testing. If we have the time to
attempt our 125 percent goal, we might try other machines with more GPU powers.
### SCHEDULE
3.24-3.30: Understand the logic of FFT; Understand research papers about parallel GPU FFT.
3.31-4.06: Write sequential code for FFT; Implement parallel CPU FFT.
4.07-4.13: Analyze parallel CPU FFT performance; Prepare for checkpoint report.
4.14-4.20: Implement parallel GPU FFT; Analyze parallel GPU FFT performance.
4.21-4.27: Understand research papers about parallel FFT’s application in image/voice processing.
4.28-5.04: Try to implement FFT’s application in image/voice processing.
### References
[1] Wikipedia. Fast Fourier transform — Wikipedia, the free encyclopedia. http://en.wikipedia.
org/w/index.php?title=Fast\%20Fourier\%20transform&oldid=1073961290, 2022. [On-
line; accessed 22-March-2022].
