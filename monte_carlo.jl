### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0a7ccbce-d228-11ec-34a3-1fc3ac51f1a8
begin
	using Random 
	using RNGPool  # For parallel random numbers
	using Sobol    # For Sobol only
	using QuasiMonteCarlo # For lattice sampline
	using HCubature, StaticArrays  # For adaptive cubature
	
	using Plots
	using PlutoUI
end

# ╔═╡ 8c48f025-cd13-4c3a-9bcf-42775ffa783b
md"""
# Lab 9: Bayesian Computing & Integration
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ 16d063b3-09e3-4dbd-852f-e5e56c4d6c0b
md"""
## Sampling Patterns in 2-d
"""

# ╔═╡ 4987b363-c8d0-4f99-b7e9-2a811d1b1956
function f_gaussian_at_origin(x; sigma::Real = 1.0) 
	result = exp(-sum(x.^2)./(2*sigma^2))
	result /= 2π*sigma^2
	return result
end

# ╔═╡ 4750c434-1d5b-48f8-bfe8-3fb087eb1127
max_evals_2d_error_plt = 2^16;

# ╔═╡ eaab2562-0081-426b-9087-524c57cd7c2c
begin  # pre-compute 2-d sampling patterns for interactive graphic below
	Random.seed!(42)
	p_uniform = QuasiMonteCarlo.sample(max_evals_2d_error_plt,zeros(2),ones(2),UniformSample())
	p_sobol = QuasiMonteCarlo.sample(max_evals_2d_error_plt,zeros(2),ones(2),SobolSample())
	p_lattice = QuasiMonteCarlo.sample(max_evals_2d_error_plt,zeros(2),ones(2),LatticeRuleSample())
	p_lds = QuasiMonteCarlo.sample(max_evals_2d_error_plt,zeros(2),ones(2),LowDiscrepancySample([10,3]))
end;

# ╔═╡ 4b078935-4ae8-4645-a676-b4d93096a0e8
max_max_evals_2d_plt = 2000;

# ╔═╡ 1d6bbc7f-707f-43cf-a09c-b636f714ee35
md"Number of samples: $(@bind max_evals_2d_plt Slider(1:max_max_evals_2d_plt; default=1))"

# ╔═╡ 8215ae2e-988f-4ed3-92d5-610d2bbc6056
md"Standard deviation of normal distribution: $(@bind sigma_sample  confirm(NumberField(0.02:0.02:1, default=0.1)))"

# ╔═╡ e650a01a-de89-464f-886e-ffdda6ad26c3
let
	ms = 1
	pltsize = (800,800)
	plt_x = plt_y = range(0, stop = 1, length = 100)
	errstr = "" #"\n(Δ = " * string(round(Δ_uniform_2d,digits=5)) * ")"
	plt1 = scatter(view(p_uniform,1,1:max_evals_2d_plt), view(p_uniform,2,1:max_evals_2d_plt), xlims=(0,1), ylims=(0,1), legend=:none, ms=ms, size=pltsize, title="Uniform sample" * errstr )
	contour!(plt_x, plt_y, (x,y) -> f_gaussian_at_origin([x,y],sigma=sigma_sample))
	
	errstr = "" #"\n(Δ = " * string(round(Δ_sobol_2d,digits=5)) * ")"
	plt2 = scatter(view(p_sobol,1,1:max_evals_2d_plt), view(p_sobol,2,1:max_evals_2d_plt), xlims=(0,1), ylims=(0,1), legend=:none, ms=ms, size=pltsize, title="Sobol sequence" * errstr )
	contour!(plt_x, plt_y, (x,y) -> f_gaussian_at_origin([x,y],sigma=sigma_sample))
	
	errstr = "" #"\n(Δ = " * string(round(Δ_lattice_2d,digits=5)) * ")"
	plt3 = scatter(view(p_lattice,1,1:max_evals_2d_plt), view(p_lattice,2,1:max_evals_2d_plt), xlims=(0,1), ylims=(0,1), legend=:none, ms=ms, size=pltsize, title="Lattice rule\n" * errstr )
	contour!(plt_x, plt_y, (x,y) -> f_gaussian_at_origin([x,y],sigma=sigma_sample))
	
	errstr = "" #"\n(Δ = " * string(round(Δ_lds_2d,digits=5)) * ")"
	plt4 = scatter(view(p_lds,1,1:max_evals_2d_plt), view(p_lds,2,1:max_evals_2d_plt), xlims=(0,1), ylims=(0,1), legend=:none, ms=ms, size=pltsize, title="Low discrepancy sequence"*errstr )
	contour!(plt_x, plt_y, (x,y) -> f_gaussian_at_origin([x,y],sigma=sigma_sample))
	plot(plt1, plt2, plt3, plt4)
end

# ╔═╡ 0fd920e1-209e-4456-9f3a-025537fe081d
md"""
**Question:**  Which of the stamling strategies above do you expect will provide the most accurate estimate of a Gaussian integral?
"""

# ╔═╡ 647d29ed-0c0d-4640-ab96-28a33f4b9814
md"""
## Integration Error in 2-d
### Normal distribution at origin
"""

# ╔═╡ 4c7942f1-ef77-460b-a40d-24a0c8961cdb
best_estimate_2d = hcubature(x->f_gaussian_at_origin(x,sigma=sigma_sample), zeros(2), ones(2), rtol=eps(Float64), atol=0, maxevals=1_000_000)[1]

# ╔═╡ f570b0d7-36cc-456d-a056-f6f2ebc25b97
#log2_max_evals_2d = 14
md"log₂ maximum evaluations: $(@bind log2_max_evals_2d NumberField(8:16, default=15))"

# ╔═╡ e89d3d18-b3a8-490b-96f1-6fdf792178ac
n_to_test_mc_2d = 2 .^(1:log2_max_evals_2d);

# ╔═╡ b3cd3ea8-7acd-4695-8521-4a1939308ef5
md"### Alternative function in 2-d"

# ╔═╡ 5fb9905a-6562-4d6d-af30-e4c9aca17723
begin
	md"Standard deviation of each Gaussian: $(@bind sigma_err confirm(NumberField(0.02:0.02:0.5, default=0.1)))"
end

# ╔═╡ d0f920e4-3448-494b-9620-3b39859a113f
md"Minimum of y-axis: $(@bind log10_y_axis_min_2d Slider(-16:-3, default=-6))"

# ╔═╡ 837fdc17-604b-4a48-8039-5a68e6a8a2df
md"""
**Question:**  How does the integration error change as you varry the standard deviation of the Gaussians in the target distribution?  

**Question:**  What are the implications of your findins for analyzing datasets with a large number of observations?
"""

# ╔═╡ 440af812-7000-4f27-9276-94a23c80b735
md"## Integration Error in Higher Dimensions"

# ╔═╡ 6bfb3b7c-5021-4910-93fb-0dc4325ce1ac
md"Redraw peak locations: $(@bind regen_multipeaks_highd Button())"

# ╔═╡ 6c776177-f29d-4fa8-83e2-6fcc1bbb542e
md"minimum y-axis: $(@bind log10_y_axis_min Slider(-16:-3, default=-6))"

# ╔═╡ 936c7b13-ae2a-432d-88b3-a7f504ffe9ba
md"""
**Question:**  How does the error of standard Monte Carlo method change as you increase the number of dimensions?  Integration using Sobol sampling?  

**Question:** H-Cubature is a type of adaptive quadrature.  How does it compare to integration via Sobol sampling for a few to several dimensions?  What about for ∼10-12 dimensions?

**Question:**  What are the implications of your findings for analyzing datasets with a large number of model parameters?
"""

# ╔═╡ fa7061f2-1126-4965-8915-60474759ff41
md"# Setup & Helper Code"

# ╔═╡ 822c11e1-c64a-4400-91ea-f78a51553216
use_threads = true

# ╔═╡ d5f5fbe6-b75b-4874-bb20-e562287ed51b
TableOfContents()

# ╔═╡ a0a1ca74-3c26-461b-8ae6-b0a4b8caa53e
nbsp = html"&nbsp;";

# ╔═╡ 3da61e81-2619-4206-be08-38ed262cb517
md"Redraw peak locations: $(@bind regen_multipeaks_2d Button()) $nbsp $nbsp "

# ╔═╡ e28ebf18-a1a3-4449-94fb-0dea42093cff
 begin 
	regen_multipeaks_2d
	const num_peaks_2d = 4
	const peak_centers_2d = rand(2, num_peaks_2d)
	function f_multipeaks_2d(x; sigma::Real ) 
		result = 0.0
		for i in 1:num_peaks_2d
			result += exp(-sum((x.-peak_centers_2d[:,i]).^2)/(2*sigma^2))
		end
		return result
	end
end 

# ╔═╡ f0f87a8d-779b-45e2-bbe9-0b0ccfbfa6d3
md"""
Function to use for evaluating: 
$(@bind func_to_integrate_2d Select([f_gaussian_at_origin => "Gaussian at origin", f_multipeaks_2d => "Multiple Peaks"]; default=f_multipeaks_2d))
"""

# ╔═╡ d8a19f6d-da16-486a-a705-b907c454b0e3
let
	x = y = range(0, stop = 1, length = 100)
	contour(x, y, (x,y) -> func_to_integrate_2d([x,y], sigma=sigma_err) , size=(400,400))
	xlabel!("x")
	ylabel!("y")
end

# ╔═╡ 82c1c073-d97a-41f5-98a1-4630fb41d095
@bind ndim_plt_param confirm(
	PlutoUI.combine() do Child
md"""   		
Number of dimensions for integrand: $(Child("num_dim", NumberField(1:12, default=2)))
$nbsp $nbsp
σ: $(Child("sigma", NumberField(0.02:0.02:1.0, default=0.2)))
$nbsp $nbsp
log₂(Max evaluations): 
$(Child("log2_max_evals", NumberField(8:17, default=14)))  
"""
	end
)

# ╔═╡ 3cc73fc1-bec4-4bfa-9094-3a114226468e
begin 
	num_dim = ndim_plt_param.num_dim
	sigma_err_highd = ndim_plt_param.sigma
	log2_max_evals = ndim_plt_param.log2_max_evals
	n_to_test_mc = 2 .^(1:log2_max_evals)
end;

# ╔═╡ 58788411-f41e-45d7-a11c-88307d82dc57
 begin 
	regen_multipeaks_highd
	const num_peaks = 4
	const peak_centers = rand(num_dim, num_peaks)
	function f_multipeaks(x; sigma::Real) 
		result = 0.0
		for i in 1:num_peaks
			result += exp(-sum((x.-peak_centers[:,i]).^2)/(2*sigma^2))
		end
		return result
	 end
end;

# ╔═╡ 0ff55d52-b17e-42a3-83d0-6cbb1a812f29
if func_to_integrate_2d == f_multipeaks_2d 
	func_to_integrate = f_multipeaks
else
	func_to_integrate = f_gaussian_at_origin
end

# ╔═╡ 1c0c9b4c-bf8e-4257-a839-e4a02542cdfd
begin
	best_estimate = hcubature(x->func_to_integrate(x,sigma=sigma_err_highd), zeros(num_dim), ones(num_dim), rtol=1e-16, atol=0.0, maxevals=1_000_000)[1]
	(;best_estimate )
end

# ╔═╡ 00b690af-20f4-43ea-afae-28d655c8ea13
begin
    estimates_hcubature_tmp = map(n->hcubature(x->func_to_integrate(x,sigma=sigma_err_highd), zeros(num_dim), ones(num_dim), rtol=eps(Float64), atol=0, maxevals=n),  n_to_test_mc)
	estimates_hcubature = map(x->x[1], estimates_hcubature_tmp)
	estimates_hcubature_error = map(x->x[2], estimates_hcubature_tmp)
end;

# ╔═╡ 59f92f75-04f0-4e1a-820e-d05a27104f25
md"## Integration routines"

# ╔═╡ c3855f10-c085-4c22-afab-20444f5c7e1e
function integrate_2d_grid_serial(f::Function, sqrt_n::Integer)
	num_dim = 2
	tmp = zeros(num_dim)
	result = 0.0
	for i in 1:sqrt_n
		x = (i-0.5)/sqrt_n
		for j in 1:sqrt_n
			y = (j-0.5)/sqrt_n
			result += f((x,y))
		end
	end
	return result / sqrt_n^2
end

# ╔═╡ bb44e7b0-2da9-4622-8f50-0fb4870188bb
function integrate_monte_carlo_serial(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	#rng = Random.seed!(seed)
	setRNGs(seed)
	rng::RNG = getRNG()	
	result = 0.0
	tmp = zeros(num_dim)
	for i in 1:n
		rand!(rng, tmp)
		result += f(tmp)
	end
	return result / n
end

# ╔═╡ e7657f38-9fbc-4949-b9a4-638fba3549c7
function integrate_monte_carlo_parallel(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	setRNGs(seed)
	num_threads = Threads.nthreads()
	result_per_thread = zeros(num_threads)
	n_per_thread = div(n, num_threads )
	Threads.@threads for t in 1:num_threads
		local rng::RNG = getRNG()	
		arg = zeros(num_dim)
        for i in 1:n_per_thread
			rand!(rng, arg)
			result_per_thread[t] += f(arg)
        end
	end
	result = sum(result_per_thread)
	if true && ((num_threads * n_per_thread)<n)
		# Add any extra itterations due to n/num_threads not being an integer
		local rng::RNG = getRNG()	
		for i in (num_threads * n_per_thread):n
			result += f(rand(rng))
		end
		result /= n
	else
		result = sum(result_per_thread) / (num_threads * n_per_thread)
	end
	return result
end

# ╔═╡ d09876c9-7181-4d58-8fff-82453e43541c
if use_threads
	estimates_mc = integrate_monte_carlo_parallel.(x->func_to_integrate(x,sigma=sigma_err_highd), n_to_test_mc,num_dim=num_dim)
else
	estimates_mc = integrate_monte_carlo_serial.(x->func_to_integrate(x,sigma=sigma_err_highd), n_to_test_mc,num_dim=num_dim)
end;

# ╔═╡ fede99f4-d0b8-469d-bbe9-4756606b6f7d
function integrate_sobol_serial(f::Function, n::Integer; num_dim::Integer = 2)
	s = SobolSeq(num_dim)
	result = 0.0
	tmp = zeros(num_dim)
	for i in 1:n
		next!(s,tmp)
		result += f(tmp)
	end
	return result / n
end

# ╔═╡ 46ce3e37-92bd-4e77-bc4a-dfb2434843b2
function integrate_sobol_parallel(f::Function, n::Integer; num_dim::Integer = 2)
	num_threads = Threads.nthreads()
	s_per_thread = [ SobolSeq(num_dim) for t in 1:num_threads ]
	result_per_thread = zeros(num_threads)
	n_per_thread = div(n, num_threads )
	#arg = zeros(num_dim, num_threads)
	Threads.@threads for t in 1:num_threads
		local s = s_per_thread[t]
		local num_to_skip = n_per_thread * (t-1)
		local arg = arg = zeros(num_dim)
		if num_to_skip >= 1
			#println("Helo from ",t)
			s = skip(s,num_to_skip, exact=true)
		end
        for i in 1:n_per_thread
			#next!(s, view(arg,:,t))
			next!(s, arg)
			#println("i = ", i, "  x = ", view(arg,:,t), " t = ", t)
			result_per_thread[t] += f(arg)
        end
	end
	result = sum(result_per_thread) 
	if ((num_threads * n_per_thread)<n)
		# Add any extra itterations due to n/num_threads not being an integer
		local s = SobolSeq(num_dim)
		local num_to_skip = n_per_thread * num_threads
		s = skip(s,num_to_skip, exact=true) 
		local arg = zeros(num_dim)
		for i in (num_threads * n_per_thread+1):n
			next!(s, arg)
			result += f(arg)
		end
		result /= n
	else
		result /= (num_threads * n_per_thread)
	end
	return result
end

# ╔═╡ 7859fa4f-3c5c-425f-9c15-ce0da7a85990
if use_threads
	estimates_sobol = integrate_sobol_parallel.(x->func_to_integrate(x,sigma=sigma_err_highd), n_to_test_mc,num_dim=num_dim) 
else
	estimates_sobol = integrate_sobol_serial.(x->func_to_integrate(x,sigma=sigma_err_highd), n_to_test_mc,num_dim=num_dim) 
end;

# ╔═╡ 9e6e6850-7320-4cfb-9528-b3467be70a69
begin
	plt = plot(yaxis=:log, legend=:bottomleft)
	
	plot!(plt,log10.(n_to_test_mc), abs.(estimates_mc .- best_estimate)./best_estimate, label="Monte Carlo", markershape=:circle, color=1)
	plot!(plt,log10.(n_to_test_mc), abs.(estimates_sobol .- best_estimate)./best_estimate, 	label="Sobol", markershape=:circle, alpha=0.5, color=2)
	plot!(plt,log10.(n_to_test_mc), abs.(estimates_hcubature .- best_estimate)./best_estimate, 
			label="H-Cubature", markershape=:circle, color=3)
	ylims!(10.0 .^ log10_y_axis_min,2)
	xlabel!("log₁₀(Number of Evaluations)")
	ylabel!("abs(Error)/(best estimate)")
	title!("Error versus Number of Evaluations:\n" * string(num_dim) * "-d Mixture of Gaussians")
	plt
end

# ╔═╡ 97f6edce-3bfd-4c53-9662-8d3304b41b7c
function integrate_uniform(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	Random.seed!(seed)
	pts =
		(num_dim==2) && (n<=size(p_uniform,2)) ? p_uniform :  # try to reuse 2-d samples
		QuasiMonteCarlo.sample(n,zeros(num_dim),ones(num_dim),UniformSample())
		
	mapreduce(i->f(view(pts,:,i)), +, 1:n)/n
end

# ╔═╡ 638a3cf6-82ba-4411-a702-d0cde64e567b
function integrate_sobol(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	Random.seed!(seed)
	pts =
		(num_dim==2) && (n<=size(p_sobol,2)) ? p_sobol :  # try to reuse 2-d samples
		QuasiMonteCarlo.sample(n,zeros(num_dim),ones(num_dim),SobolSample())
		
	mapreduce(i->f(view(pts,:,i)), +, 1:n)/n
end

# ╔═╡ 96713364-ebb1-4b9b-bb5e-0ff9d14bc1cf
function integrate_lattice(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	Random.seed!(seed)
	pts =
		(num_dim==2) && (n<=size(p_lattice,2)) ? p_lattice :  # try to reuse 2-d samples
		QuasiMonteCarlo.sample(n,zeros(num_dim),ones(num_dim),LatticeRuleSample())
		
	mapreduce(i->f(view(pts,:,i)), +, 1:n)/n
end

# ╔═╡ 4332698d-3357-405a-8ad9-3bef0ed5ebfd
function integrate_lds(f::Function, n::Integer; seed::Integer = 42, num_dim::Integer = 2)
	@assert 1 <= num_dim <= 10 
	base = [10,3,7,11,13,17,19,23,31,37][1:num_dim]
	Random.seed!(seed)
	pts =
		(num_dim==2) && (n<=size(p_lds,2)) ? p_lds :  # try to reuse 2-d samples
		QuasiMonteCarlo.sample(n,zeros(num_dim),ones(num_dim),LowDiscrepancySample(base))
	mapreduce(i->f(view(pts,:,i)), +, 1:n)/n
end

# ╔═╡ 6fcce823-2e56-488a-b9e3-bb40aadb11e0
begin 
	Δgauss_uniform_2d(n::Integer) = (integrate_uniform(x->f_gaussian_at_origin(x,sigma=sigma_sample), n)  - best_estimate_2d) / best_estimate_2d
	Δgauss_sobol_2d(n::Integer) = (integrate_sobol(x->f_gaussian_at_origin(x,sigma=sigma_sample), n)  - best_estimate_2d) / best_estimate_2d
	Δgauss_lattice_2d(n::Integer) = (integrate_lattice(x->f_gaussian_at_origin(x,sigma=sigma_sample), n)  - best_estimate_2d) / best_estimate_2d
	Δgauss_lds_2d(n::Integer) = (integrate_lds(x->f_gaussian_at_origin(x,sigma=sigma_sample), n)  - best_estimate_2d) / best_estimate_2d
end;

# ╔═╡ be807633-68c7-4b9a-91f7-a8cdaee56814
(;Δ_uniform = Δgauss_uniform_2d(max_evals_2d_plt), Δ_sobol=Δgauss_sobol_2d(max_evals_2d_plt), Δ_lattice=Δgauss_lattice_2d(max_evals_2d_plt), Δ_lds = Δgauss_lds_2d(max_evals_2d_plt))

# ╔═╡ 05fd15ad-5082-4091-bdf6-1c492c18b09d
begin 
	y_plt_gauss_2d_uniform = abs.(Δgauss_uniform_2d.(n_to_test_mc_2d))
	y_plt_gauss_2d_sobol = abs.(Δgauss_sobol_2d.(n_to_test_mc_2d))
	y_plt_gauss_2d_lattice = abs.(Δgauss_lattice_2d.(n_to_test_mc_2d))
	y_plt_gauss_2d_lds = abs.(Δgauss_lds_2d.(n_to_test_mc_2d))
end;

# ╔═╡ 98e1c267-f13e-4e43-92a7-5e3384ed7c53
let
	plt = plot(yaxis=:log, legend=:bottomleft)
	
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_gauss_2d_uniform, label="Monte Carlo", markershape=:circle, color=1)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_gauss_2d_sobol, 	label="Sobol", markershape=:circle, alpha=0.5, color=2)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_gauss_2d_lattice, 	label="Lattice", markershape=:circle, alpha=0.5, color=3)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_gauss_2d_lds, 	label="LDS", markershape=:circle, alpha=0.5, color=4)
	#plot!(plt,log10.(n_to_test_mc), abs.(estimates_hcubature .- best_estimate)./best_estimate, label="H-Cubature", markershape=:circle, color=5)
	ylims!(10.0 .^ log10_y_axis_min,2)
	xlabel!("log₁₀(Number of Evaluations)")
	ylabel!("abs(Error)/(best estimate)")
	title!("Error versus Number of Evaluations: 2-d Gaussian")
	plt
end

# ╔═╡ 87a5ee82-0287-4003-8d18-d626fd48c7dc
begin 
	best_estimate_alt_2d = hcubature(x->func_to_integrate_2d(x,sigma=sigma_err), zeros(2), ones(2), rtol=eps(Float64), atol=0, maxevals=1_000_000)[1]
	Δuser_uniform_2d(n::Integer) = (integrate_uniform(x->func_to_integrate_2d(x,sigma=sigma_err), n)  - best_estimate_alt_2d) / best_estimate_alt_2d
	Δuser_sobol_2d(n::Integer) = (integrate_sobol(x->func_to_integrate_2d(x,sigma=sigma_err), n)  - best_estimate_alt_2d) / best_estimate_alt_2d
	Δuser_lattice_2d(n::Integer) = (integrate_lattice(x->func_to_integrate_2d(x,sigma=sigma_err), n)  - best_estimate_alt_2d) / best_estimate_alt_2d
	Δuser_lds_2d(n::Integer) = (integrate_lds(x->func_to_integrate_2d(x,sigma=sigma_err), n)  - best_estimate_alt_2d) / best_estimate_alt_2d
	#Δuser_hcubature_2d(n::Integer) = (hcubature(x->func_to_integrate_2d(x,sigma=sigma_err), zeros(2), ones(2), rtol=eps(Float64), atol=0, maxevals=n)[1]  - best_estimate_alt_2d) / best_estimate_alt_2d
end;

# ╔═╡ 39b132ea-e49c-4641-8293-55f59ee649df
begin  # Precompute values for next plot
	y_plt_user_2d_uniform = abs.(Δuser_uniform_2d.(n_to_test_mc_2d))
	y_plt_user_2d_sobol = abs.(Δuser_sobol_2d.(n_to_test_mc_2d))
	y_plt_user_2d_lattice = abs.(Δuser_lattice_2d.(n_to_test_mc_2d))
	y_plt_user_2d_lds = abs.(Δuser_lds_2d.(n_to_test_mc_2d))
end;

# ╔═╡ 44dc0410-eef2-4cba-b5d5-a1d849650d47
let
	plt = plot(yaxis=:log, legend=:bottomleft)
	
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_user_2d_uniform, label="Monte Carlo", markershape=:circle, color=1)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_user_2d_sobol, 	label="Sobol", markershape=:circle, alpha=0.5, color=2)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_user_2d_lattice, 	label="Lattice", markershape=:circle, alpha=0.5, color=3)
	plot!(plt,log10.(n_to_test_mc_2d), y_plt_user_2d_lds, 	label="LDS", markershape=:circle, alpha=0.5, color=4)
	#plot!(plt,log10.(n_to_test_mc_2d), abs.(Δuser_hcubature_2d.(n_to_test_mc_2d)), label="H-Cubature", markershape=:circle, color=5)
	ylims!(10.0 .^ log10_y_axis_min_2d,2)
	xlabel!("log₁₀(Number of Evaluations)")
	ylabel!("abs(Error)/(best estimate)")
	title!("Error versus Number of Evaluations:\n2-d Mixture of Gaussians")
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HCubature = "19dc6840-f33b-545b-b366-655c7e3ffd49"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuasiMonteCarlo = "8a4e6c94-4038-4cdc-81c3-7e6ffdb2a71b"
RNGPool = "c7fc2d14-d53c-5e81-ac30-66aba9c03525"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Sobol = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
HCubature = "~1.5.0"
Plots = "~1.29.0"
PlutoUI = "~0.7.38"
QuasiMonteCarlo = "~0.2.4"
RNGPool = "~2.0.0"
Sobol = "~1.5.0"
StaticArrays = "~1.4.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "134af3b940d1ca25b19bc9740948157cee7ff8fa"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.5.0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[deps.LatticeRules]]
deps = ["Random"]
git-tree-sha1 = "7f5b02258a3ca0221a6a9710b0a0a2e8fb4957fe"
uuid = "73f95e8e-ec14-4e6a-8b18-0d2e271c4e55"
version = "0.0.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.QuasiMonteCarlo]]
deps = ["Distributions", "LatinHypercubeSampling", "LatticeRules", "Sobol"]
git-tree-sha1 = "bc69c718a83951dcb999404ff267a7b8c39c1c63"
uuid = "8a4e6c94-4038-4cdc-81c3-7e6ffdb2a71b"
version = "0.2.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.RNGPool]]
deps = ["Random123"]
git-tree-sha1 = "a8b726e04b6942d2e4b70f1b76f97a173c9ed993"
uuid = "c7fc2d14-d53c-5e81-ac30-66aba9c03525"
version = "2.0.0"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═8c48f025-cd13-4c3a-9bcf-42775ffa783b
# ╟─16d063b3-09e3-4dbd-852f-e5e56c4d6c0b
# ╟─4987b363-c8d0-4f99-b7e9-2a811d1b1956
# ╟─4750c434-1d5b-48f8-bfe8-3fb087eb1127
# ╟─eaab2562-0081-426b-9087-524c57cd7c2c
# ╟─e650a01a-de89-464f-886e-ffdda6ad26c3
# ╟─1d6bbc7f-707f-43cf-a09c-b636f714ee35
# ╟─4b078935-4ae8-4645-a676-b4d93096a0e8
# ╟─8215ae2e-988f-4ed3-92d5-610d2bbc6056
# ╟─0fd920e1-209e-4456-9f3a-025537fe081d
# ╟─be807633-68c7-4b9a-91f7-a8cdaee56814
# ╟─647d29ed-0c0d-4640-ab96-28a33f4b9814
# ╟─98e1c267-f13e-4e43-92a7-5e3384ed7c53
# ╟─4c7942f1-ef77-460b-a40d-24a0c8961cdb
# ╟─6fcce823-2e56-488a-b9e3-bb40aadb11e0
# ╟─f570b0d7-36cc-456d-a056-f6f2ebc25b97
# ╟─e89d3d18-b3a8-490b-96f1-6fdf792178ac
# ╟─05fd15ad-5082-4091-bdf6-1c492c18b09d
# ╟─b3cd3ea8-7acd-4695-8521-4a1939308ef5
# ╟─e28ebf18-a1a3-4449-94fb-0dea42093cff
# ╟─f0f87a8d-779b-45e2-bbe9-0b0ccfbfa6d3
# ╟─d8a19f6d-da16-486a-a705-b907c454b0e3
# ╟─5fb9905a-6562-4d6d-af30-e4c9aca17723
# ╟─3da61e81-2619-4206-be08-38ed262cb517
# ╟─87a5ee82-0287-4003-8d18-d626fd48c7dc
# ╟─39b132ea-e49c-4641-8293-55f59ee649df
# ╟─44dc0410-eef2-4cba-b5d5-a1d849650d47
# ╟─d0f920e4-3448-494b-9620-3b39859a113f
# ╟─837fdc17-604b-4a48-8039-5a68e6a8a2df
# ╟─440af812-7000-4f27-9276-94a23c80b735
# ╟─82c1c073-d97a-41f5-98a1-4630fb41d095
# ╟─6bfb3b7c-5021-4910-93fb-0dc4325ce1ac
# ╟─1c0c9b4c-bf8e-4257-a839-e4a02542cdfd
# ╟─9e6e6850-7320-4cfb-9528-b3467be70a69
# ╟─6c776177-f29d-4fa8-83e2-6fcc1bbb542e
# ╟─936c7b13-ae2a-432d-88b3-a7f504ffe9ba
# ╟─3cc73fc1-bec4-4bfa-9094-3a114226468e
# ╟─58788411-f41e-45d7-a11c-88307d82dc57
# ╟─0ff55d52-b17e-42a3-83d0-6cbb1a812f29
# ╟─d09876c9-7181-4d58-8fff-82453e43541c
# ╟─7859fa4f-3c5c-425f-9c15-ce0da7a85990
# ╟─00b690af-20f4-43ea-afae-28d655c8ea13
# ╟─fa7061f2-1126-4965-8915-60474759ff41
# ╠═0a7ccbce-d228-11ec-34a3-1fc3ac51f1a8
# ╟─822c11e1-c64a-4400-91ea-f78a51553216
# ╟─d5f5fbe6-b75b-4874-bb20-e562287ed51b
# ╟─a0a1ca74-3c26-461b-8ae6-b0a4b8caa53e
# ╟─59f92f75-04f0-4e1a-820e-d05a27104f25
# ╟─c3855f10-c085-4c22-afab-20444f5c7e1e
# ╟─bb44e7b0-2da9-4622-8f50-0fb4870188bb
# ╟─e7657f38-9fbc-4949-b9a4-638fba3549c7
# ╟─fede99f4-d0b8-469d-bbe9-4756606b6f7d
# ╟─46ce3e37-92bd-4e77-bc4a-dfb2434843b2
# ╟─97f6edce-3bfd-4c53-9662-8d3304b41b7c
# ╟─638a3cf6-82ba-4411-a702-d0cde64e567b
# ╟─96713364-ebb1-4b9b-bb5e-0ff9d14bc1cf
# ╟─4332698d-3357-405a-8ad9-3bef0ed5ebfd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
