import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import csv
from pathlib import Path

"""
Gate                         |-------Rgd1--------|                                         Drain
------Lg---Rg----------------|                   |-------------------------Rg----Ld--------
    |           |            |-----Cgd---Rgd2----|     |       |      |               |
    |           |                                      |       |      |               |                        
    |           |                                      |       |      |               |
    Cpg         |                                      |       |      |               Cpd
    |       ------------                               |       |      |               |
    |       |          |+                              |      Rds    Cds              |
    |       |         Cgs   Vg      Gm*Vg*exp(-1j * w * Td)    |      |               |
    |       |          |-                              |       |      |               |
    |      Rgs1        |                               |       |      |               |
    |       |          |                               |       |      |               |
    |       |         Rgs2                             |       |      |               |
    |       |          |                               |       |      |               |
    |       -----------------------------------------------------------               |
    |                                  |                                              |
    |                                  |                                              |
    |                                  |                                              |
    |                                  Rs                                             |
    |                                  |                                              |
    |                                  |                                              |
    |                                  Ls                                             |
    |                                  |                                              |
    |                                  |                                              |
    |----------------------------------------------------------------------------------
                                       |
                                       |
                                       |
                                       |
                                     Source       
"""


FALLBACK_DEFAULT_PARAMS = {
    'Lg': 8e-11,
    'Rg': 0.5,
    'Cpg': 4.0e-12,
    'Rd': 1.32,
    'Ld': 9.9e-11,
    'Cpd': 4.22e-13,
    'Rs': 0.08,
    'Ls': 1e-12,
    'Rgs1': 1e5/5.1,
    'Cgs': 3.99e-12,
    'Rgs2': 2.34,
    'Rgd1': 1e7/2.1,
    'Cgd': 2.47e-13,
    'Rgd2': 10.5,
    'Cds': 6.67e-13,
    'Rds': 1000/8.97,
    'Gm': 0.681,
    'Td': 3.91e-12,
}


def load_default_params_from_data_csv(data_csv_path=None):
    if data_csv_path is None:
        data_csv_path = Path(__file__).with_name('Data.csv')

    raw_values = {}
    with open(data_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if not key or key.startswith('------'):
                continue
            try:
                raw_values[key] = float(value)
            except ValueError:
                continue

    direct_keys = ['Lg', 'Rg', 'Cpg', 'Rd', 'Ld', 'Cpd', 'Rs', 'Ls', 'Cgs', 'Cgd']
    required_derived_keys = ['Ggs', 'Rgs', 'Ggd', 'Rgd', 'Cds_intrinsic', 'gds', 'gm', 'Tgm']

    missing_direct = [k for k in direct_keys if k not in raw_values]
    missing_derived = [k for k in required_derived_keys if k not in raw_values]
    missing_all = missing_direct + missing_derived
    if missing_all:
        raise KeyError(f"Missing keys in {data_csv_path}: {', '.join(missing_all)}")

    params = {k: raw_values[k] for k in direct_keys}
    params['Rgs1'] = 1.0 / raw_values['Ggs']
    params['Rgs2'] = raw_values['Rgs']
    params['Rgd1'] = 1.0 / raw_values['Ggd']
    params['Rgd2'] = raw_values['Rgd']
    params['Cds'] = raw_values['Cds_intrinsic']
    params['Rds'] = 1.0 / raw_values['gds']
    params['Gm'] = raw_values['gm']
    params['Td'] = raw_values['Tgm']
    return params


try:
    DEFAULT_PARAMS = load_default_params_from_data_csv()
    print('Data.csv is used.')
except FileNotFoundError:
    DEFAULT_PARAMS = FALLBACK_DEFAULT_PARAMS.copy()
    print('Fallback default parameters are used.')

TRACE_COLORS = {
    'measured': 'black',
    'sim_initial': 'tab:blue',
    'sim_optimized': 'tab:orange',
    'sim_best': 'tab:orange',
    'sim_generic': 'tab:green',
}

PARAM_BOUNDS = {
    'Lg': (5e-12, 4e-10),
    'Rg': (0.01, 20.0),
    'Cpg': (1e-14, 5e-11),
    'Rd': (0.01, 30.0),
    'Ld': (5e-12, 4e-10),
    'Cpd': (1e-14, 2e-11),
    'Rs': (1e-3, 5.0),
    'Ls': (1e-13, 2e-10),
    'Rgs1': (100.0, 1e7),
    'Cgs': (1e-14, 1e-10),
    'Rgs2': (1e-2, 1e4),
    'Rgd1': (1e3, 1e9),
    'Cgd': (1e-14, 1e-10),
    'Rgd2': (1e-2, 1e5),
    'Cds': (1e-14, 1e-10),
    'Rds': (1.0, 1e5),
    'Gm': (1e-3, 5.0),
    'Td': (1e-14, 2e-11),
}


def build_node_index():
    return {
        'gate': 0,
        'drain': 1,
        'M1_g_pkg': 2,
        'M1_g_int': 3,
        'M1_gs_mid': 4,
        'M1_gd_mid': 5,
        'M1_d_pkg': 6,
        'M1_d_int': 7,
        'M1_s_pkg': 8,
        'M1_s_int': 9,
    }


def stamp_admittance(Y, n1, n2, value):
    Y[n1, n1] += value
    Y[n2, n2] += value
    Y[n1, n2] -= value
    Y[n2, n1] -= value


def stamp_to_ground(Y, n, value):
    Y[n, n] += value


def stamp_vccs(Y, n_plus, n_minus, c_plus, c_minus, gm_value):
    Y[n_plus, c_plus] += gm_value
    Y[n_plus, c_minus] -= gm_value
    Y[n_minus, c_plus] -= gm_value
    Y[n_minus, c_minus] += gm_value


def build_Y_matrix(freq_hz, params, node):
    w = 2 * np.pi * freq_hz
    jw = 1j * w
    Y = np.zeros((10, 10), dtype=complex)

    y_lg = 1 / (jw * params['Lg'])
    y_rg = 1 / params['Rg']
    y_cpg = jw * params['Cpg']

    y_rd = 1 / params['Rd']
    y_ld = 1 / (jw * params['Ld'])
    y_cpd = jw * params['Cpd']

    y_rs = 1 / params['Rs']
    y_ls = 1 / (jw * params['Ls'])

    y_rgs1 = 1 / params['Rgs1']
    y_cgs = jw * params['Cgs']
    y_rgs2 = 1 / params['Rgs2']

    y_rgd1 = 1 / params['Rgd1']
    y_cgd = jw * params['Cgd']
    y_rgd2 = 1 / params['Rgd2']

    y_cds = jw * params['Cds']
    y_rds = 1 / params['Rds']

    gm_eff = params['Gm'] * np.exp(-1j * w * params['Td'])

    stamp_admittance(Y, node['gate'], node['M1_g_pkg'], y_lg)
    stamp_admittance(Y, node['M1_g_pkg'], node['M1_g_int'], y_rg)
    stamp_to_ground(Y, node['M1_g_pkg'], y_cpg)

    stamp_admittance(Y, node['drain'], node['M1_d_pkg'], y_rd)
    stamp_admittance(Y, node['M1_d_pkg'], node['M1_d_int'], y_ld)
    stamp_to_ground(Y, node['M1_d_pkg'], y_cpd)

    stamp_to_ground(Y, node['M1_s_pkg'], y_rs)
    stamp_admittance(Y, node['M1_s_pkg'], node['M1_s_int'], y_ls)

    stamp_admittance(Y, node['M1_g_int'], node['M1_s_int'], y_rgs1)
    stamp_admittance(Y, node['M1_g_int'], node['M1_gs_mid'], y_cgs)
    stamp_admittance(Y, node['M1_gs_mid'], node['M1_s_int'], y_rgs2)

    stamp_admittance(Y, node['M1_g_int'], node['M1_d_int'], y_rgd1)
    stamp_admittance(Y, node['M1_g_int'], node['M1_gd_mid'], y_cgd)
    stamp_admittance(Y, node['M1_gd_mid'], node['M1_d_int'], y_rgd2)

    stamp_admittance(Y, node['M1_d_int'], node['M1_s_int'], y_cds)
    stamp_admittance(Y, node['M1_d_int'], node['M1_s_int'], y_rds)

    stamp_vccs(
        Y,
        node['M1_d_int'],
        node['M1_s_int'],
        node['M1_g_int'],
        node['M1_s_int'],
        gm_eff,
    )

    return Y


def extract_two_port_Y(freq_hz, params, node):
    Yfull = build_Y_matrix(freq_hz, params, node)

    port_idx = [node['gate'], node['drain']]
    int_idx = [i for i in range(10) if i not in port_idx]

    Ypp = Yfull[np.ix_(port_idx, port_idx)]
    Ypi = Yfull[np.ix_(port_idx, int_idx)]
    Yip = Yfull[np.ix_(int_idx, port_idx)]
    Yii = Yfull[np.ix_(int_idx, int_idx)]

    return Ypp - Ypi @ np.linalg.solve(Yii, Yip)


def y_to_s(y2, z0=50.0):
    I = np.eye(2, dtype=complex)
    z0m = z0 * I
    return (I - z0m @ y2) @ np.linalg.inv(I + z0m @ y2)


def simulate_s_parameters(freq_hz, params, z0=50.0):
    node = build_node_index()
    s = np.zeros((len(freq_hz), 2, 2), dtype=complex)

    for i, f in enumerate(freq_hz):
        y2 = extract_two_port_Y(f, params, node)
        s[i] = y_to_s(y2, z0=z0)

    return s


def normalized_error(sim_s, target_s):
    denom = np.maximum(np.abs(target_s), 1e-3)
    residual = np.abs(sim_s - target_s) / denom
    return float(np.sqrt(np.mean(residual ** 2)))


def normalized_error_by_sparam(sim_s, target_s):
    s_map = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1),
    }
    errors = {}
    for name, (m, n) in s_map.items():
        sim_part = sim_s[:, m, n]
        target_part = target_s[:, m, n]
        denom = np.maximum(np.abs(target_part), 1e-3)
        residual = np.abs(sim_part - target_part) / denom
        errors[name] = float(np.sqrt(np.mean(residual ** 2)))
    return errors


def _format_freq_ghz(freq_hz):
    return f'{freq_hz / 1e9:.3f} GHz'


def annotate_smith_start_end(ax, freq_hz, s_complex, trace_label, color):
    if len(freq_hz) == 0 or len(s_complex) == 0:
        return

    start_point = s_complex[0]
    end_point = s_complex[-1]

    ax.plot(
        start_point.real,
        start_point.imag,
        marker='o',
        markersize=4,
        linestyle='None',
        color=color,
        zorder=5,
    )
    ax.plot(
        end_point.real,
        end_point.imag,
        marker='s',
        markersize=4,
        linestyle='None',
        color=color,
        zorder=5,
    )

    ax.annotate(
        f'{trace_label} start {_format_freq_ghz(freq_hz[0])}',
        xy=(start_point.real, start_point.imag),
        xytext=(4, 4),
        textcoords='offset points',
        fontsize=6,
        color=color,
        zorder=6,
    )
    ax.annotate(
        f'{trace_label} end {_format_freq_ghz(freq_hz[-1])}',
        xy=(end_point.real, end_point.imag),
        xytext=(4, -8),
        textcoords='offset points',
        fontsize=6,
        color=color,
        zorder=6,
    )


def classify_trace_label(label):
    stem = Path(str(label)).stem.lower()
    if 'sim_initial' in stem:
        return 'sim_initial'
    if 'sim_optimized' in stem:
        return 'sim_optimized'
    if 'sim' in stem:
        return 'sim_generic'
    return 'measured'


def get_trace_style(label, live=False):
    kind = classify_trace_label(label)
    if live and kind == 'sim_generic':
        kind = 'sim_best'
    color = TRACE_COLORS.get(kind, 'tab:blue')
    is_experimental = kind == 'measured'
    return color, is_experimental


def plot_smith_trace(ntw, m, n, ax, label, color, is_experimental=False, r=None):
    plot_kwargs = {
        'ax': ax,
        'label': label,
        'color': color,
    }
    if is_experimental:
        plot_kwargs.update(
            {
                'linestyle': 'None',
                'marker': 'o',
                'markersize': 2.5,
            }
        )

    if r is None:
        ntw.plot_s_smith(m=m, n=n, **plot_kwargs)
    else:
        ntw.plot_s_smith(m=m, n=n, r=r, **plot_kwargs)


def optimize_parameters(
    measured_ntw,
    initial_params,
    param_bounds,
    max_freq_ghz=10,
    max_iter=220,
    objective_decimation=4,
    seed=7,
    show_live_smith=True,
    live_plot_every=5,
):
    measured_band = measured_ntw[f'0-{max_freq_ghz}ghz']
    freq_full = measured_band.f
    s_target_full = measured_band.s

    freq_obj = freq_full[::objective_decimation]
    s_target_obj = s_target_full[::objective_decimation]

    keys = list(initial_params.keys())
    lower = np.array([param_bounds[k][0] for k in keys], dtype=float)
    upper = np.array([param_bounds[k][1] for k in keys], dtype=float)

    current = np.array([initial_params[k] for k in keys], dtype=float)
    current = np.clip(current, lower, upper)

    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    log_current = np.log10(current)

    rng = np.random.default_rng(seed)
    sparam_index = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1),
    }
    objective_weights = {
        'S11': 1.8,
        'S12': 0.7,
        'S21': 1.9,
        'S22': 1.0,
    }

    def vector_to_params(vec):
        return {k: float(v) for k, v in zip(keys, vec)}

    def weighted_error(sim_s, target_s):
        weighted_sum = 0.0
        total_weight = 0.0
        for name, (m, n) in sparam_index.items():
            sim_part = sim_s[:, m, n]
            target_part = target_s[:, m, n]
            denom = np.maximum(np.abs(target_part), 1e-3)
            residual = np.abs(sim_part - target_part) / denom
            err = float(np.sqrt(np.mean(residual ** 2)))
            w = float(objective_weights.get(name, 1.0))
            weighted_sum += w * (err ** 2)
            total_weight += w
        if total_weight <= 0:
            return np.inf
        return float(np.sqrt(weighted_sum / total_weight))

    def objective(vec_log):
        vec = 10 ** vec_log
        params = vector_to_params(vec)
        try:
            sim_s = simulate_s_parameters(freq_obj, params)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return np.inf
        if not np.all(np.isfinite(sim_s)):
            return np.inf
        return weighted_error(sim_s, s_target_obj)

    fig = None
    axs = None
    sim_label = 'Simulated (best)'
    measured_label = 'Measured'

    def render_live_smith(sim_s_full, iter_idx, best_err):
        nonlocal fig, axs

        if fig is None:
            plt.ion()
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        sim_ntw = rf.Network(
            frequency=rf.Frequency.from_f(freq_full, unit='Hz'),
            s=sim_s_full,
            z0=50,
        )

        s21_mag_max = float(
            np.nanmax(
                [
                    np.nanmax(np.abs(measured_band.s[:, 1, 0])),
                    np.nanmax(np.abs(sim_ntw.s[:, 1, 0])),
                ]
            )
        )
        s21_radius = max(1.0, float(np.ceil(s21_mag_max * 1.05 * 10) / 10))

        for ax in axs.flat:
            ax.cla()

        measured_color, measured_is_exp = get_trace_style(measured_label, live=True)
        sim_color, sim_is_exp = get_trace_style(sim_label, live=True)

        plot_smith_trace(
            measured_band,
            m=0,
            n=0,
            ax=axs[0, 0],
            label=measured_label,
            color=measured_color,
            is_experimental=measured_is_exp,
        )
        annotate_smith_start_end(
            axs[0, 0],
            measured_band.f,
            measured_band.s[:, 0, 0],
            measured_label,
            measured_color,
        )
        plot_smith_trace(
            sim_ntw,
            m=0,
            n=0,
            ax=axs[0, 0],
            label=sim_label,
            color=sim_color,
            is_experimental=sim_is_exp,
        )
        annotate_smith_start_end(
            axs[0, 0],
            sim_ntw.f,
            sim_ntw.s[:, 0, 0],
            sim_label,
            sim_color,
        )
        axs[0, 0].set_title('S11')
        axs[0, 0].legend(loc='best', fontsize=8)

        measured_s12_scaled = measured_band.copy()
        measured_s12_scaled.s[:, 0, 1] = 20 * measured_s12_scaled.s[:, 0, 1]
        sim_s12_scaled = sim_ntw.copy()
        sim_s12_scaled.s[:, 0, 1] = 20 * sim_s12_scaled.s[:, 0, 1]
        plot_smith_trace(
            measured_s12_scaled,
            m=0,
            n=1,
            ax=axs[0, 1],
            label=measured_label,
            color=measured_color,
            is_experimental=measured_is_exp,
        )
        annotate_smith_start_end(
            axs[0, 1],
            measured_s12_scaled.f,
            measured_s12_scaled.s[:, 0, 1],
            measured_label,
            measured_color,
        )
        plot_smith_trace(
            sim_s12_scaled,
            m=0,
            n=1,
            ax=axs[0, 1],
            label=sim_label,
            color=sim_color,
            is_experimental=sim_is_exp,
        )
        annotate_smith_start_end(
            axs[0, 1],
            sim_s12_scaled.f,
            sim_s12_scaled.s[:, 0, 1],
            sim_label,
            sim_color,
        )
        axs[0, 1].set_title('20×S12')
        axs[0, 1].legend(loc='best', fontsize=8)

        plot_smith_trace(
            measured_band,
            m=1,
            n=0,
            r=s21_radius,
            ax=axs[1, 0],
            label=measured_label,
            color=measured_color,
            is_experimental=measured_is_exp,
        )
        annotate_smith_start_end(
            axs[1, 0],
            measured_band.f,
            measured_band.s[:, 1, 0],
            measured_label,
            measured_color,
        )
        plot_smith_trace(
            sim_ntw,
            m=1,
            n=0,
            r=s21_radius,
            ax=axs[1, 0],
            label=sim_label,
            color=sim_color,
            is_experimental=sim_is_exp,
        )
        annotate_smith_start_end(
            axs[1, 0],
            sim_ntw.f,
            sim_ntw.s[:, 1, 0],
            sim_label,
            sim_color,
        )
        axs[1, 0].set_title(f'S21 (auto r={s21_radius:.1f})')
        axs[1, 0].legend(loc='best', fontsize=8)

        plot_smith_trace(
            measured_band,
            m=1,
            n=1,
            ax=axs[1, 1],
            label=measured_label,
            color=measured_color,
            is_experimental=measured_is_exp,
        )
        annotate_smith_start_end(
            axs[1, 1],
            measured_band.f,
            measured_band.s[:, 1, 1],
            measured_label,
            measured_color,
        )
        plot_smith_trace(
            sim_ntw,
            m=1,
            n=1,
            ax=axs[1, 1],
            label=sim_label,
            color=sim_color,
            is_experimental=sim_is_exp,
        )
        annotate_smith_start_end(
            axs[1, 1],
            sim_ntw.f,
            sim_ntw.s[:, 1, 1],
            sim_label,
            sim_color,
        )
        axs[1, 1].set_title('S22')
        axs[1, 1].legend(loc='best', fontsize=8)

        fig.suptitle(
            f'Optimization live Smith chart (up to {max_freq_ghz} GHz) | '
            f'iter {iter_idx}/{max_iter} | best error {best_err:.6f}',
            fontsize=12,
            fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw_idle()
        plt.pause(0.001)

    pop_size = max(24, 2 * len(keys))
    mutation_factor = 0.75
    crossover_rate = 0.85
    polish_every = 8
    polish_factors = np.array([0.80, 0.90, 0.96, 1.04, 1.10, 1.22], dtype=float)

    population = np.empty((pop_size, len(keys)), dtype=float)
    population[0] = log_current.copy()
    if pop_size > 1:
        population[1:] = rng.uniform(log_lower, log_upper, size=(pop_size - 1, len(keys)))

    scores = np.array([objective(population[i]) for i in range(pop_size)], dtype=float)
    best_idx = int(np.argmin(scores))
    best_log = population[best_idx].copy()
    best_obj = float(scores[best_idx])
    history = [best_obj]

    if show_live_smith:
        try:
            initial_sim_full = simulate_s_parameters(freq_full, vector_to_params(10 ** best_log))
            render_live_smith(initial_sim_full, iter_idx=0, best_err=best_obj)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            show_live_smith = False

    for iter_idx in range(max_iter):
        for i in range(pop_size):
            choices = np.delete(np.arange(pop_size), i)
            a_idx, b_idx, c_idx = rng.choice(choices, size=3, replace=False)
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            mutant = np.clip(a + mutation_factor * (b - c), log_lower, log_upper)
            cross_mask = rng.random(len(keys)) < crossover_rate
            force_idx = int(rng.integers(0, len(keys)))
            cross_mask[force_idx] = True
            trial = np.where(cross_mask, mutant, population[i])

            trial_obj = objective(trial)
            if trial_obj < scores[i]:
                population[i] = trial
                scores[i] = trial_obj

                if trial_obj < best_obj:
                    best_obj = float(trial_obj)
                    best_log = trial.copy()

        if (iter_idx + 1) % polish_every == 0:
            local_best = best_log.copy()
            local_best_obj = best_obj
            base_vec = 10 ** best_log

            for param_idx in range(len(keys)):
                for fac in polish_factors:
                    candidate_vec = base_vec.copy()
                    candidate_vec[param_idx] = np.clip(
                        candidate_vec[param_idx] * fac,
                        lower[param_idx],
                        upper[param_idx],
                    )
                    candidate_log = np.log10(candidate_vec)
                    candidate_obj = objective(candidate_log)
                    if candidate_obj < local_best_obj:
                        local_best_obj = float(candidate_obj)
                        local_best = candidate_log

            if local_best_obj < best_obj:
                best_obj = local_best_obj
                best_log = local_best
                worst_idx = int(np.argmax(scores))
                population[worst_idx] = best_log.copy()
                scores[worst_idx] = best_obj

        history.append(best_obj)

        if show_live_smith and (iter_idx + 1) % max(1, int(live_plot_every)) == 0:
            try:
                best_sim_full = simulate_s_parameters(freq_full, vector_to_params(10 ** best_log))
                render_live_smith(best_sim_full, iter_idx=iter_idx + 1, best_err=best_obj)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                pass

    if show_live_smith and max_iter % max(1, int(live_plot_every)) != 0:
        try:
            best_sim_full = simulate_s_parameters(freq_full, vector_to_params(10 ** best_log))
            render_live_smith(best_sim_full, iter_idx=max_iter, best_err=best_obj)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            pass

    if show_live_smith:
        plt.ioff()

    best_params = vector_to_params(10 ** best_log)
    sim_initial = simulate_s_parameters(freq_full, initial_params)
    sim_best = simulate_s_parameters(freq_full, best_params)

    initial_by_s = normalized_error_by_sparam(sim_initial, s_target_full)
    optimized_by_s = normalized_error_by_sparam(sim_best, s_target_full)

    result = {
        'initial_error': normalized_error(sim_initial, s_target_full),
        'optimized_error': normalized_error(sim_best, s_target_full),
        'initial_error_by_s': initial_by_s,
        'optimized_error_by_s': optimized_by_s,
        'history': history,
        'frequency_hz': freq_full,
        'sim_initial_s': sim_initial,
        'sim_best_s': sim_best,
    }
    return best_params, result


def generate_simulated_s2p(out_basename='circuit_2_delay_gm_0p5_50GHz', params=None, freq=None):
    params = DEFAULT_PARAMS.copy() if params is None else dict(params)

    if freq is None:
        f_start = 0.5e9
        f_stop = 10e9
        n_points = 501
        freq = np.linspace(f_start, f_stop, n_points)

    s = simulate_s_parameters(freq, params)

    ntw = rf.Network(frequency=rf.Frequency.from_f(freq, unit='Hz'), s=s, z0=50)
    ntw.write_touchstone(out_basename)

    out_file_with_ext = Path(f'{out_basename}.s2p')
    out_file_no_ext = Path(out_basename)
    if out_file_no_ext.exists():
        out_file_no_ext.replace(out_file_with_ext)
    out_file = str(out_file_with_ext)

    print('Simulated S2P generation complete')
    print(f"Td = {params['Td']:.3e} s, Gm = {params['Gm']:.3f} S")
    print(f'Generated: {out_file}')

    return out_file

def write_s2p_from_s(out_basename, freq, s, z0=50):
    ntw = rf.Network(frequency=rf.Frequency.from_f(freq, unit='Hz'), s=s, z0=z0)
    ntw.write_touchstone(out_basename)

    out_file_with_ext = Path(f'{out_basename}.s2p')
    out_file_no_ext = Path(out_basename)
    if out_file_no_ext.exists():
        out_file_no_ext.replace(out_file_with_ext)
    out_file = str(out_file_with_ext)

    return out_file


def save_optimization_csv(initial_params, optimized_params, optimization, out_csv='optimization_report.csv'):
    def fmt_sci(value):
        if value == '':
            return ''
        value = float(value)
        if not np.isfinite(value):
            return ''
        return f'{value:.2e}'

    headers = [
        'category',
        'name',
        'initial_value',
        'optimized_value',
        'percent_change',
        'initial_error',
        'optimized_error',
        'error_improvement_percent',
    ]

    rows = []

    for name in initial_params:
        initial_value = float(initial_params[name])
        optimized_value = float(optimized_params[name])

        if abs(initial_value) > 0:
            percent_change = 100.0 * (optimized_value - initial_value) / abs(initial_value)
        else:
            percent_change = np.nan

        rows.append({
            'category': 'parameter',
            'name': name,
            'initial_value': fmt_sci(initial_value),
            'optimized_value': fmt_sci(optimized_value),
            'percent_change': fmt_sci(percent_change),
            'initial_error': '',
            'optimized_error': '',
            'error_improvement_percent': '',
        })

    for name in ['S11', 'S12', 'S21', 'S22']:
        initial_error = float(optimization['initial_error_by_s'][name])
        optimized_error = float(optimization['optimized_error_by_s'][name])
        if initial_error > 0:
            error_improvement = 100.0 * (initial_error - optimized_error) / initial_error
        else:
            error_improvement = np.nan

        rows.append({
            'category': 'sparameter_error',
            'name': name,
            'initial_value': '',
            'optimized_value': '',
            'percent_change': '',
            'initial_error': fmt_sci(initial_error),
            'optimized_error': fmt_sci(optimized_error),
            'error_improvement_percent': fmt_sci(error_improvement),
        })

    overall_initial = float(optimization['initial_error'])
    overall_optimized = float(optimization['optimized_error'])
    if overall_initial > 0:
        overall_improvement = 100.0 * (overall_initial - overall_optimized) / overall_initial
    else:
        overall_improvement = np.nan

    rows.append({
        'category': 'overall_error',
        'name': 'ALL',
        'initial_value': '',
        'optimized_value': '',
        'percent_change': '',
        'initial_error': fmt_sci(overall_initial),
        'optimized_error': fmt_sci(overall_optimized),
        'error_improvement_percent': fmt_sci(overall_improvement),
    })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return out_csv


def plot_smith_comparison(filenames, max_freq_ghz=10, out_file='circuit_2_delay_compare_smith.png'):
    ntws = [rf.Network(filename) for filename in filenames]
    ntws = [ntw[f'0-{max_freq_ghz}ghz'] for ntw in ntws]

    s21_mag_max = np.nanmax([np.nanmax(np.abs(ntw.s[:, 1, 0])) for ntw in ntws])
    s21_radius = max(1.0, float(np.ceil(s21_mag_max * 1.05 * 10) / 10))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f'S-parameters comparison on Smith Chart (up to {max_freq_ghz} GHz)',
        fontsize=14,
        fontweight='bold'
    )

    trace_items = []
    for ntw, label in zip(ntws, filenames):
        color, is_experimental = get_trace_style(label, live=False)
        trace_items.append((ntw, label, color, is_experimental))

    for ntw, label, color, is_experimental in trace_items:
        plot_smith_trace(ntw, m=0, n=0, ax=axs[0, 0], label=label, color=color, is_experimental=is_experimental)
        annotate_smith_start_end(axs[0, 0], ntw.f, ntw.s[:, 0, 0], label, color)
    axs[0, 0].set_title('S11')
    axs[0, 0].legend(loc='best', fontsize=8)

    for ntw, label, color, is_experimental in trace_items:
        ntw_s12_scaled = ntw.copy()
        ntw_s12_scaled.s[:, 0, 1] = 20 * ntw_s12_scaled.s[:, 0, 1]
        plot_smith_trace(
            ntw_s12_scaled,
            m=0,
            n=1,
            ax=axs[0, 1],
            label=label,
            color=color,
            is_experimental=is_experimental,
        )
        annotate_smith_start_end(axs[0, 1], ntw_s12_scaled.f, ntw_s12_scaled.s[:, 0, 1], label, color)
    axs[0, 1].set_title('20×S12')
    axs[0, 1].legend(loc='best', fontsize=8)

    for ntw, label, color, is_experimental in trace_items:
        plot_smith_trace(
            ntw,
            m=1,
            n=0,
            r=s21_radius,
            ax=axs[1, 0],
            label=label,
            color=color,
            is_experimental=is_experimental,
        )
        annotate_smith_start_end(axs[1, 0], ntw.f, ntw.s[:, 1, 0], label, color)
    axs[1, 0].set_title(f'S21 (auto r={s21_radius:.1f})')
    axs[1, 0].legend(loc='best', fontsize=8)

    for ntw, label, color, is_experimental in trace_items:
        plot_smith_trace(ntw, m=1, n=1, ax=axs[1, 1], label=label, color=color, is_experimental=is_experimental)
        annotate_smith_start_end(axs[1, 1], ntw.f, ntw.s[:, 1, 1], label, color)
    axs[1, 1].set_title('S22')
    axs[1, 1].legend(loc='best', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=300)
    plt.show()

    print(f'Smith chart (up to {max_freq_ghz} GHz) saved: {out_file}')


def main():
    measured_file = 'deviceVg-2.75VD28.s2p'
    measured_base = measured_file.rsplit('.', 1)[0]
    measured_ntw = rf.Network(measured_file)
    measured_band = measured_ntw['0-10ghz']

    initial_params = DEFAULT_PARAMS.copy()

    simulated_initial_file = generate_simulated_s2p(
        f'{measured_base}_sim_initial',
        params=initial_params,
        freq=measured_band.f,
    )

    best_params, optimization = optimize_parameters(
        measured_ntw=measured_ntw,
        initial_params=initial_params,
        param_bounds=PARAM_BOUNDS,
        max_freq_ghz=10,
        max_iter=220,
        objective_decimation=4,
        seed=7,
        show_live_smith=True,
        live_plot_every=5,
    )

    simulated_optimized_file = write_s2p_from_s(
        f'{measured_base}_sim_optimized',
        optimization['frequency_hz'],
        optimization['sim_best_s'],
        z0=50,
    )

    report_csv = save_optimization_csv(
        initial_params=initial_params,
        optimized_params=best_params,
        optimization=optimization,
        out_csv=f'{measured_base}_opt_rep.csv',
    )

    print('\nOptimization summary')
    print(f"Initial normalized error  : {optimization['initial_error']:.6f}")
    print(f"Optimized normalized error: {optimization['optimized_error']:.6f}")
    print('Per-parameter normalized errors (initial -> optimized):')
    for name in ['S11', 'S12', 'S21', 'S22']:
        e0 = optimization['initial_error_by_s'][name]
        e1 = optimization['optimized_error_by_s'][name]
        print(f'  {name}: {e0:.6f} -> {e1:.6f}')
    if optimization['initial_error'] > 0:
        imp = 100.0 * (optimization['initial_error'] - optimization['optimized_error']) / optimization['initial_error']
        print(f'Improvement               : {imp:.2f}%')
    print(f'Optimization report saved to: {report_csv}')

    plot_smith_comparison(
        [simulated_initial_file, simulated_optimized_file, measured_file],
        max_freq_ghz=10,
        out_file=f'{measured_base}_compare_smith_optimized.png',
    )


if __name__ == '__main__':
    main()
