using XLSX
using PyPlot
using DataFrames

function parseXLSX(filename::String)
    time = Float64[]
    acceleration = Float64[]

    XLSX.openxlsx(filename) do xf
        sheet = xf[1]
        row = 2
        while true
            t_cell = sheet[row, 1]
            a_cell = sheet[row, 2]

            if (t_cell === nothing || t_cell === missing) &&
               (a_cell === nothing || a_cell === missing)
                break
            end

            if t_cell !== nothing && t_cell !== missing &&
               a_cell !== nothing && a_cell !== missing
                push!(time, Float64(t_cell))
                push!(acceleration, Float64(a_cell))
            end

            row += 1
        end
    end

    return time, acceleration
end

struct PlotSpec{T<:AbstractFloat}
    dim::Int
    x_axis::Vector{T}
    y_axis::Vector{T}
    title::String
    z_data::Union{Matrix{T}, Nothing}
    xstart::Union{Nothing, Real}
    xend::Union{Nothing, Real}
    ystart::Union{Nothing, Real}
    yend::Union{Nothing, Real}
    zstart::Union{Nothing, Real}
    zend::Union{Nothing, Real}
    xlabel::String
    ylabel::String
    zlabel::String
    grid::Bool
    figsize::Tuple{Real, Real}
    plot_type::Union{Nothing, Symbol}
    highlight_zero_y::Bool
    min_range_x::Union{Nothing, Real}
    min_range_y::Union{Nothing, Real}
    min_range_z::Union{Nothing, Real}
    display_x_0::Bool
    display_y_0::Bool
    display_z_0::Bool

    function PlotSpec(
        dim::Int,
        x_axis::Vector{T},
        y_axis::Vector{T};
        title::String = "Graph",
        z_data::Union{Matrix{T}, Nothing} = nothing,
        xstart::Union{Nothing, Real} = nothing,
        xend::Union{Nothing, Real} = nothing,
        ystart::Union{Nothing, Real} = nothing,
        yend::Union{Nothing, Real} = nothing,
        zstart::Union{Nothing, Real} = nothing,
        zend::Union{Nothing, Real} = nothing,
        xlabel::String = "X Axis",
        ylabel::String = "Y Axis",
        zlabel::String = "Y Axis",
        grid::Bool = true,
        figsize::Tuple{Real, Real} = (5, 4),
        plot_type::Union{Nothing, Symbol}=nothing,
        highlight_zero_y::Bool=false,
        min_range_x::Union{Nothing, Real}=nothing,
        min_range_y::Union{Nothing, Real}=nothing,
        min_range_z::Union{Nothing, Real}=nothing,
        display_x_0::Bool=false,
        display_y_0::Bool=false,
        display_z_0::Bool=false,
    ) where {T<:AbstractFloat}

        new{T}(dim, x_axis, y_axis, title, z_data,
               xstart, xend, ystart, yend, zstart, zend,
               xlabel, ylabel, zlabel, grid, figsize, plot_type, highlight_zero_y,
               min_range_x, min_range_y, min_range_z, display_x_0, display_y_0, display_z_0)
    end
end

# DATAFRAME UTILITY
function extract_number(str::String)
    m = match(r"[-+]?[0-9]*\.?[0-9]+", str)
    return m === nothing ? error("No number found in string: $str") : parse(Float64, m.match)
end

function dataframe_to_heatmap(df::DataFrame)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}
    # Extract Y axis values (assumed numeric or convertible)
    y_raw = df[!, 1]
    
    # Extract X axis names (strings)
    x_raw = names(df)[2:end]
    x = extract_number.(x_raw)

    # Extract data matrix with possible missing values
    data_with_missing = Matrix(df[:, 2:end])

    # Replace missing with NaN
    data = replace(data_with_missing, missing => NaN)

    return x, y_raw, data
end

# PLOT UTILITY FUNCTIONS
function set_labels_and_limits!(ax, spec::PlotSpec; is3d=false)
    ax.set_title(spec.title)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    if is3d
        ax.set_zlabel(spec.zlabel)
    end

    # === X AXIS ===
    if spec.xstart !== nothing && spec.xend !== nothing
        xmin, xmax = spec.xstart, spec.xend
    else
        xmin, xmax = extrema(spec.x_axis)
        range = xmax - xmin

        if spec.min_range_x !== nothing && range < spec.min_range_x
            center = (xmin + xmax) / 2
            half_range = spec.min_range_x / 2
            xmin, xmax = center - half_range, center + half_range
        else
            padding = 0.1 * range
            xmin, xmax = xmin - padding, xmax + padding
        end

        if spec.display_x_0
            xmin = min(xmin, 0.0)
            xmax = max(xmax, 0.0)
            if xmin == 0.0
                xmax *= 1.1
            elseif xmax == 0.0
                xmin *= 1.1
            end
        end
    end
    ax.set_xlim(xmin, xmax)

    # === Y AXIS ===
    if spec.ystart !== nothing && spec.yend !== nothing
        ymin, ymax = spec.ystart, spec.yend
    else
        ymin, ymax = extrema(spec.y_axis)
        range = ymax - ymin

        if spec.min_range_y !== nothing && range < spec.min_range_y
            center = (ymin + ymax) / 2
            half_range = spec.min_range_y / 2
            ymin, ymax = center - half_range, center + half_range
        else
            padding = 0.1 * range
            ymin, ymax = ymin - padding, ymax + padding
        end

        if spec.display_y_0
            ymin = min(ymin, 0.0)
            ymax = max(ymax, 0.0)
            if ymin == 0.0
                ymax *= 1.1
            elseif ymax == 0.0
                ymin *= 1.1
            end
        end
    end
    ax.set_ylim(ymin, ymax)

    # === Z AXIS (only for 3D) ===
    if is3d && spec.z_data !== nothing
        if spec.zstart !== nothing && spec.zend !== nothing
            zmin, zmax = spec.zstart, spec.zend
        else
            zmin, zmax = extrema(spec.z_data)
            range = zmax - zmin

            if spec.min_range_z !== nothing && range < spec.min_range_z
                center = (zmin + zmax) / 2
                half_range = spec.min_range_z / 2
                zmin, zmax = center - half_range, center + half_range
            else
                padding = 0.1 * range
                zmin, zmax = zmin - padding, zmax + padding
            end

            if spec.display_z_0
                zmin = min(zmin, 0.0)
                zmax = max(zmax, 0.0)
                if zmin == 0.0
                    zmax *= 1.1
                elseif zmax == 0.0
                    zmin *= 1.1
                end
            end
        end
        ax.set_zlim(zmin, zmax)
    end
end

function set_grid!(ax, spec::PlotSpec)
    ax.set_axisbelow(true)
    ax.grid(spec.grid)
end

function plot_2d!(ax, spec::PlotSpec)
    if spec.plot_type == :scatter
        if spec.highlight_zero_y
            # Identify points where y ≈ 0.0
            idx_zero = findall(y -> isapprox(y, 0.0; atol=1e-8), spec.y_axis)
            idx_nonzero = setdiff(1:length(spec.y_axis), idx_zero)

            # Collect vectors to avoid slicing issues
            x_zero = collect(spec.x_axis[idx_zero])
            y_zero = collect(spec.y_axis[idx_zero])
            x_nonzero = collect(spec.x_axis[idx_nonzero])
            y_nonzero = collect(spec.y_axis[idx_nonzero])

            # Plot non-zero points normally
            ax.scatter(x_nonzero, y_nonzero, label="Trim")

            # Plot zero-y points in red
            if !isempty(idx_zero)
                ax.scatter(x_zero, y_zero, c="red", label="No Trim")  # <-- fix here
            end

            ax.legend()

        else
            ax.scatter(spec.x_axis, spec.y_axis)
        end

    elseif spec.plot_type == :heatmap
        if spec.z_data === nothing
            error("Heatmap plot requested but z_data is nothing.")
        end

        c = ax.imshow(spec.z_data;
                      extent=[minimum(spec.x_axis), maximum(spec.x_axis),
                              minimum(spec.y_axis), maximum(spec.y_axis)],
                      origin="lower", aspect="auto", cmap="viridis")
        cb = colorbar(c, ax=ax)
        cb.set_label(spec.zlabel)

    else
        ax.plot(spec.x_axis, spec.y_axis)
    end

    set_labels_and_limits!(ax, spec)
    set_grid!(ax, spec)
end

function plot_3d!(fig, ax, spec::PlotSpec)
    if spec.z_data === nothing
        error("3D plot requested but z_data is nothing.")
    end

    x, y, Z = spec.x_axis, spec.y_axis, spec.z_data
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))

    surf = ax.plot_surface(X, Y, Z, cmap="viridis")
    set_labels_and_limits!(ax, spec; is3d=true)
    set_grid!(ax, spec)
    fig.colorbar(surf, ax=ax)
end

# PLOTTING
function plot_spec(spec::PlotSpec)
    if spec.dim == 2
        figure(figsize=spec.figsize)
        ax = gca()
        plot_2d!(ax, spec)
        show()

    elseif spec.dim == 3
        fig = figure(figsize=spec.figsize)
        ax = fig.add_subplot(111, projection="3d")
        plot_3d!(fig, ax, spec)
        show()

    else
        error("Unsupported plot dimension: $(spec.dim). Only dim=2 or dim=3 are supported.")
    end
end

function plot_specs_grid(specs::Vector{<:PlotSpec})
    plots_per_window = 4
    n = length(specs)
    num_windows = ceil(Int, n / plots_per_window)

    for win in 1:num_windows
        start_idx = (win - 1) * plots_per_window + 1
        end_idx = min(win * plots_per_window, n)
        current_specs = specs[start_idx:end_idx]

        fig, axs = subplots(2, 2, figsize=(10, 8))
        axs = vec(axs)

        for (i, spec) in enumerate(current_specs)
            ax = axs[i]

            if spec.dim == 2
                plot_2d!(ax, spec)

            elseif spec.dim == 3
                fig.delaxes(ax)
                ax3d = fig.add_subplot(2, 2, i, projection="3d")
                plot_3d!(fig, ax3d, spec)

            else
                error("Unsupported dimension: $(spec.dim)")
            end
        end

        # Remove unused axes
        for i in (length(current_specs)+1):4
            fig.delaxes(axs[i])
        end

        tight_layout(pad=3.0)
        fig.subplots_adjust(wspace=0.4, hspace=0.6)
        show()
    end
end

function compute_velocity(times::Vector{Float64}, accels::Vector{Float64}; v0::Float64=1.8)
    n = length(times)
    velocities = Float64[v0]

    for i in 2:n
        dt = times[i] - times[i-1]
        avg_accel = (accels[i] + accels[i-1]) / 2
        new_velocity = velocities[end] + avg_accel * dt
        push!(velocities, new_velocity)
    end

    return velocities
end

times, accels = parseXLSX("Vehicle Impact - raw acceleration data.xlsx")

A = PlotSpec(2, times, accels; xlabel="Time (s)", ylabel="Acceleration (m/s/s)", title="Accleration vs. Time of Impact", display_x_0=true)
vels = compute_velocity(times, accels)
V = PlotSpec(2, times, vels; xlabel="Times (s)", ylabel="Velocity (m/s)", title="Velocity vs. Time of Impact", display_x_0=true)
mass = 0.4
forces = accels .* mass
F = PlotSpec(2, times, forces; xlabel="Times (s)", ylabel="Force (N)", title="Force vs. Time of Impact", display_x_0=true)
plot_specs_grid([A, V, F])