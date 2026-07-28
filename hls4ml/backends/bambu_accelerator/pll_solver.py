def solve_pll(
    input_freq_mhz: float, targets: list[float], use_external_oscillator: bool = False, hdl: str = 'verilog'
) -> str:
    """
    Solves the PLL configuration problem using Google OR-Tools.

    Args:
        input_freq_mhz (float): The input frequency in MHz.
        targets (list[float]): A list of target frequencies in MHz.
        use_external_oscillator (bool): Whether to use an external oscillator (default: False).
        hdl (str): Output HDL language, 'verilog' or 'vhdl' (default: 'verilog').

    Returns:
        str: Generated HDL code string if a valid configuration is found, otherwise an error message.
    """
    from ortools.sat.python import cp_model  # deferred: only PLL generation needs ortools

    # --- 1. HARDWARE CONSTANTS ---
    PFD_MIN, PFD_MAX = 10.0, 50.0
    VCO_MIN, VCO_MAX = 300.0, 800.0
    MAX_PLLS = 7

    # Divider Maps
    # Dynamic (5 slots): Ratios -> Code
    dyn_ratios = {
        2: 0,
        4: 1,
        6: 2,
        8: 3,
        10: 4,
        20: 5,
        40: 6,
        60: 7,
        80: 8,
        100: 9,
        200: 10,
        400: 11,
        600: 12,
        800: 13,
        1000: 14,
        2000: 15,
    }

    # Static (1 slot each): Ratio -> Code
    # S1..S4
    static_maps = [
        {(2 * i + 3): i for i in range(8)},  # S1
        {(2 * i + 5): i for i in range(8)},  # S2
        {(2 * i + 7): i for i in range(8)},  # S3
        {(2 * i + 9): i for i in range(8)},  # S4
    ]

    print(f'--- Optimizing for: {targets} MHz ---')

    # --- 2. GENERATE CANDIDATE VCOs ---
    # Find all VCOs that can generate at least one target
    candidate_vcos = {}  # vco_freq -> {ref, fbk, pfd}

    # Pre-calculate valid divider ratios
    valid_ratios = set(dyn_ratios.keys())
    for m in static_maps:
        valid_ratios.update(m.keys())

    # Brute force valid VCOs (filtered by target feasibility)
    for t in targets:
        feasible = False
        for r in valid_ratios:
            vco = t * r
            if VCO_MIN <= vco <= VCO_MAX:
                # Check if generatable from input
                # Try Ref Divs 1..32
                for ref_val in range(32):
                    ref_div = ref_val + 1
                    pfd = input_freq_mhz / ref_div
                    if PFD_MIN <= pfd <= PFD_MAX:
                        # Check multiplier
                        # vco = pfd * 2 * (fbk+1)
                        mult = vco / pfd
                        # Check if mult is even integer (approx)
                        if abs(mult % 2) < 1e-5 or abs(mult % 2 - 2) < 1e-5:
                            k = int(round(mult / 2))
                            fbk_val = k - 1
                            if 0 <= fbk_val <= 127:
                                # Found valid VCO
                                if vco not in candidate_vcos:
                                    candidate_vcos[vco] = {'ref': ref_val, 'fbk': fbk_val, 'pfd': pfd}
                                feasible = True
                                break  # Found one config for this VCO, sufficient
        if not feasible:
            raise Exception(
                f'Hardware cannot generate {t} MHz. Check that the target frequency has correct precision. '
                'We check for tol < 1e-5, i.e. 33.33332 < 100.0/3.0 < 33.33334.'
            )

    vco_list = sorted(candidate_vcos.keys())
    print(f'Search Space: {len(vco_list)} Candidate VCO frequencies')

    if not vco_list:
        raise Exception('Hardware cannot generate these frequencies.')

    # --- 3. CP-SAT MODEL ---
    model = cp_model.CpModel()

    # Variables
    # x[p, v]: PLL p uses VCO v
    x = {}
    for p in range(MAX_PLLS):
        for v_idx, _vco in enumerate(vco_list):
            x[p, v_idx] = model.NewBoolVar(f'x_{p}_{v_idx}')

    # assign[t, p, v]: Target t assigned to PLL p on VCO v
    assign = {}
    for t_idx in range(len(targets)):
        for p in range(MAX_PLLS):
            for v_idx in range(len(vco_list)):
                assign[t_idx, p, v_idx] = model.NewBoolVar(f'assign_{t_idx}_{p}_{v_idx}')

    # Port allocation variables: use_S1[t,p,v], use_Dyn[t,p,v]...
    # We simplify: For a specific (t, p, v) assignment, we must pick a valid port type.
    use_dyn = {}
    use_stat = {}  # Key: (t, p, v, s_idx) s_idx 0..3

    for t_idx, t in enumerate(targets):
        for p in range(MAX_PLLS):
            for v_idx, vco in enumerate(vco_list):
                ratio = int(round(vco / t))
                if ratio == 0:
                    continue
                # Validate that this ratio actually produces the target frequency
                if abs(vco / ratio - t) > 1e-5:
                    continue

                # Create Bool vars for allocation if ratio valid
                # Dynamic
                if ratio in dyn_ratios:
                    use_dyn[t_idx, p, v_idx] = model.NewBoolVar(f'dyn_{t_idx}_{p}_{v_idx}')

                # Static
                for s_idx in range(4):
                    if ratio in static_maps[s_idx]:
                        use_stat[t_idx, p, v_idx, s_idx] = model.NewBoolVar(f'stat_{s_idx}_{t_idx}_{p}_{v_idx}')

    # --- CONSTRAINTS ---

    # 1. Coverage: Each target assigned exactly once
    for t_idx in range(len(targets)):
        model.Add(sum(assign[t_idx, p, v] for p in range(MAX_PLLS) for v in range(len(vco_list))) == 1)

    # 2. PLL Configuration: Max 1 VCO per PLL
    for p in range(MAX_PLLS):
        model.Add(sum(x[p, v] for v in range(len(vco_list))) <= 1)

    # 3. Link Assignment to Configuration
    for t_idx in range(len(targets)):
        for p in range(MAX_PLLS):
            for v_idx in range(len(vco_list)):
                # If target assigned to (p,v), PLL p MUST use v
                model.Add(assign[t_idx, p, v_idx] <= x[p, v_idx])

                # 4. Link Assignment to Ports
                # assign[t,p,v] == use_dyn + sum(use_stat)
                port_vars = []
                if (t_idx, p, v_idx) in use_dyn:
                    port_vars.append(use_dyn[t_idx, p, v_idx])
                for s_idx in range(4):
                    if (t_idx, p, v_idx, s_idx) in use_stat:
                        port_vars.append(use_stat[t_idx, p, v_idx, s_idx])

                if not port_vars:
                    # Ratio invalid for this VCO -> Force 0
                    model.Add(assign[t_idx, p, v_idx] == 0)
                else:
                    model.Add(assign[t_idx, p, v_idx] == sum(port_vars))

    # 5. Port Capacity Constraints
    for p in range(MAX_PLLS):
        for v_idx in range(len(vco_list)):
            # Max 5 Dynamic per PLL/VCO
            dyn_vars = [use_dyn[t, p, v_idx] for t in range(len(targets)) if (t, p, v_idx) in use_dyn]
            model.Add(sum(dyn_vars) <= 5)

            # Max 1 per Static slot per PLL/VCO
            for s_idx in range(4):
                stat_vars = [use_stat[t, p, v_idx, s_idx] for t in range(len(targets)) if (t, p, v_idx, s_idx) in use_stat]
                model.Add(sum(stat_vars) <= 1)

    # --- OBJECTIVE ---
    # Minimize sum of active PLLs
    # Active PLL = Sum of x[p,v] across all v (since max 1 v per p)
    pll_active_vars = []
    for p in range(MAX_PLLS):
        is_active = model.NewBoolVar(f'active_{p}')
        model.Add(sum(x[p, v] for v in range(len(vco_list))) == is_active)
        pll_active_vars.append(is_active)

    model.Minimize(sum(pll_active_vars))

    # --- SOLVE ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.OPTIMAL:
            print(f'Optimal Solution Found: {solver.ObjectiveValue()} PLL(s) required. ')
        else:
            print(f'Feasible Solution Found: {solver.ObjectiveValue()} PLL(s) required.')
        if hdl == 'vhdl':
            return generate_vhdl_ortools(
                solver,
                x,
                assign,
                use_stat,
                use_dyn,
                targets,
                vco_list,
                candidate_vcos,
                dyn_ratios,
                static_maps,
                use_external_oscillator,
            )
        else:
            return generate_verilog_ortools(
                solver,
                x,
                assign,
                use_stat,
                use_dyn,
                targets,
                vco_list,
                candidate_vcos,
                dyn_ratios,
                static_maps,
                use_external_oscillator,
            )
    else:
        raise Exception('No valid configuration found.')


def generate_vhdl_ortools(
    solver,
    x,
    assign,
    use_stat,
    use_dyn,
    targets,
    vco_list,
    vco_configs,
    dyn_ratios,
    static_maps,
    use_external_oscillator=False,
):
    """
    Generates VHDL code based on the solved PLL configuration.

    Args:
        solver (cp_model.CpSolver): The solver instance containing the solution.
        x (dict): Dictionary of PLL-VCO assignment variables.
        assign (dict): Dictionary of Target-PLL-VCO assignment variables.
        use_stat (dict): Dictionary of static port usage variables.
        use_dyn (dict): Dictionary of dynamic port usage variables.
        targets (list[float]): List of target frequencies.
        vco_list (list[float]): List of candidate VCO frequencies.
        vco_configs (dict): Dictionary mapping VCO frequencies to their configuration (ref, fbk, pfd).
        dyn_ratios (dict): Dictionary mapping dynamic divider ratios to configuration codes.
        static_maps (list[dict]): Per static port, a dict mapping static divider ratios to configuration codes.
        use_external_oscillator (bool): Whether to use an external oscillator (default: False).

    Returns:
        str: A string containing the generated VHDL component instantiation for the PLLs.
    """
    vhdl = ''
    MAX_PLLS = 7

    pll_counter = 1

    for p in range(MAX_PLLS):
        # Find active VCO
        active_v_idx = -1
        for v_idx in range(len(vco_list)):
            if solver.Value(x[p, v_idx]) == 1:
                active_v_idx = v_idx
                break

        if active_v_idx == -1:
            continue  # PLL unused

        vco_freq = vco_list[active_v_idx]
        cfg = vco_configs[vco_freq]

        # Gather Assignments
        # We need to map target -> Specific Port Name & Code
        g = {}
        # Init defaults
        for i in range(1, 5):
            g[f'clk_outdiv{i}'] = (0, 3)
        for i in range(1, 6):
            g[f'clk_outdivd{i}'] = (0, 4)
        ports_map = {}
        for i in range(1, 5):
            ports_map[f'CLK_DIV{i}'] = 'open'
        for i in range(1, 6):
            ports_map[f'CLK_DIVD{i}'] = 'open'

        covered_targets = []
        used_dyn_slots = []

        for t_idx, t in enumerate(targets):
            # Check if assigned here
            if solver.Value(assign[t_idx, p, active_v_idx]) == 0:
                continue

            covered_targets.append(t)
            ratio = int(round(vco_freq / t))

            # Determine which port was selected by solver
            port_found = False

            # Check Static
            for s_idx in range(4):
                if (t_idx, p, active_v_idx, s_idx) in use_stat:
                    if solver.Value(use_stat[t_idx, p, active_v_idx, s_idx]) == 1:
                        # Assigned to Static s_idx+1
                        port = f'CLK_DIV{s_idx + 1}'
                        code = static_maps[s_idx][ratio]
                        g[f'clk_outdiv{s_idx + 1}'] = (code, 3)
                        ports_map[port] = f'clk_{int(t)}mhz'
                        port_found = True
                        break

            # Check Dynamic
            if not port_found and (t_idx, p, active_v_idx) in use_dyn:
                if solver.Value(use_dyn[t_idx, p, active_v_idx]) == 1:
                    # Assigned to Dynamic. Find first free slot.
                    # Since solver guaranteed count <= 5, we just greedy fill slots.
                    for d in range(1, 6):
                        if d not in used_dyn_slots:
                            used_dyn_slots.append(d)
                            port = f'CLK_DIVD{d}'
                            code = dyn_ratios[ratio]
                            g[f'clk_outdivd{d}'] = (code, 4)
                            ports_map[port] = f'clk_{str(t).replace(".", "_")}mhz'
                            port_found = True
                            break

        # Append VHDL Block
        vhdl += f"""
    -- PLL {pll_counter}: VCO={vco_freq:.1f}MHz (PFD={cfg['pfd']:.1f}MHz)
    -- Generates: {covered_targets} MHz
    PLL_{pll_counter}: NX_PLL_U
    generic map (
        location => "", -- default location
        ref_osc_on => '{'1' if not use_external_oscillator else '0'}',
        use_pll => '1',
        ext_fbk_on => '0', -- use internal feedback
        fbk_delay_on => '0',
        fbk_delay => to_bitvector(conv_std_logic_vector(0,6)),

        -- ref_intdiv register = divide ratio (hardware-validated golden: 375/15 = 25 MHz PFD);
        -- cfg['ref'] stores ratio-1 internally, so emit +1.
        ref_intdiv   => to_bitvector(conv_std_logic_vector({cfg['ref'] + 1},5)),
        fbk_intdiv   => to_bitvector(conv_std_logic_vector({cfg['fbk']},7)),

        clk_outdiv1  => to_bitvector(conv_std_logic_vector({g['clk_outdiv1'][0]},{g['clk_outdiv1'][1]})),
        clk_outdiv2  => to_bitvector(conv_std_logic_vector({g['clk_outdiv2'][0]},{g['clk_outdiv2'][1]})),
        clk_outdiv3  => to_bitvector(conv_std_logic_vector({g['clk_outdiv3'][0]},{g['clk_outdiv3'][1]})),
        clk_outdiv4  => to_bitvector(conv_std_logic_vector({g['clk_outdiv4'][0]},{g['clk_outdiv4'][1]})),

        clk_outdivd1 => to_bitvector(conv_std_logic_vector({g['clk_outdivd1'][0]},{g['clk_outdivd1'][1]})),
        clk_outdivd2 => to_bitvector(conv_std_logic_vector({g['clk_outdivd2'][0]},{g['clk_outdivd2'][1]})),
        clk_outdivd3 => to_bitvector(conv_std_logic_vector({g['clk_outdivd3'][0]},{g['clk_outdivd3'][1]})),
        clk_outdivd4 => to_bitvector(conv_std_logic_vector({g['clk_outdivd4'][0]},{g['clk_outdivd4'][1]})),
        clk_outdivd5 => to_bitvector(conv_std_logic_vector({g['clk_outdivd5'][0]},{g['clk_outdivd5'][1]}))
    )
    port map (
        REF => '{'0' if not use_external_oscillator else 'ref_clk'}',
        FBK => '0', -- use internal feedback
        R => rst, -- active high reset
        VCO => open,
        LDFO => open,
        REFO => open,
        OSC => open, -- optionally connect to get the internal oscillator
        CAL_LOCKED => open,
        PLL_LOCKED => locked_{pll_counter},
        CLK_DIV1   => {ports_map['CLK_DIV1']},
        CLK_DIV2   => {ports_map['CLK_DIV2']},
        CLK_DIV3   => {ports_map['CLK_DIV3']},
        CLK_DIV4   => {ports_map['CLK_DIV4']},
        CLK_DIVD1  => {ports_map['CLK_DIVD1']},
        CLK_DIVD2  => {ports_map['CLK_DIVD2']},
        CLK_DIVD3  => {ports_map['CLK_DIVD3']},
        CLK_DIVD4  => {ports_map['CLK_DIVD4']},
        CLK_DIVD5  => {ports_map['CLK_DIVD5']}
    );
"""
        pll_counter += 1

    return vhdl


def generate_verilog_ortools(
    solver,
    x,
    assign,
    use_stat,
    use_dyn,
    targets,
    vco_list,
    vco_configs,
    dyn_ratios,
    static_maps,
    use_external_oscillator=False,
):
    """
    Generates Verilog code based on the solved PLL configuration.

    Args: same as generate_vhdl_ortools.

    Returns:
        str: A string containing the generated Verilog instantiation for the PLLs.
    """
    verilog = ''
    MAX_PLLS = 7

    pll_counter = 1

    for p in range(MAX_PLLS):
        # Find active VCO
        active_v_idx = -1
        for v_idx in range(len(vco_list)):
            if solver.Value(x[p, v_idx]) == 1:
                active_v_idx = v_idx
                break

        if active_v_idx == -1:
            continue  # PLL unused

        vco_freq = vco_list[active_v_idx]
        cfg = vco_configs[vco_freq]

        # Gather assignments
        g = {}
        for i in range(1, 5):
            g[f'clk_outdiv{i}'] = (0, 3)
        for i in range(1, 6):
            g[f'clk_outdivd{i}'] = (0, 4)
        ports_map = {}
        for i in range(1, 5):
            ports_map[f'CLK_DIV{i}'] = '()'
        for i in range(1, 6):
            ports_map[f'CLK_DIVD{i}'] = '()'

        covered_targets = []
        used_dyn_slots = []

        for t_idx, t in enumerate(targets):
            if solver.Value(assign[t_idx, p, active_v_idx]) == 0:
                continue

            covered_targets.append(t)
            ratio = int(round(vco_freq / t))

            port_found = False

            # Check Static
            for s_idx in range(4):
                if (t_idx, p, active_v_idx, s_idx) in use_stat:
                    if solver.Value(use_stat[t_idx, p, active_v_idx, s_idx]) == 1:
                        port = f'CLK_DIV{s_idx + 1}'
                        code = static_maps[s_idx][ratio]
                        g[f'clk_outdiv{s_idx + 1}'] = (code, 3)
                        ports_map[port] = f'(clk_{int(t)}mhz)'
                        port_found = True
                        break

            # Check Dynamic
            if not port_found and (t_idx, p, active_v_idx) in use_dyn:
                if solver.Value(use_dyn[t_idx, p, active_v_idx]) == 1:
                    for d in range(1, 6):
                        if d not in used_dyn_slots:
                            used_dyn_slots.append(d)
                            port = f'CLK_DIVD{d}'
                            code = dyn_ratios[ratio]
                            g[f'clk_outdivd{d}'] = (code, 4)
                            ports_map[port] = f'(clk_{str(t).replace(".", "_")}mhz)'
                            port_found = True
                            break

        ref_conn = "1'b0" if not use_external_oscillator else 'ref_clk'
        ref_osc = "1'b1" if not use_external_oscillator else "1'b0"

        verilog += f"""
// PLL {pll_counter}: VCO={vco_freq:.1f}MHz (PFD={cfg['pfd']:.1f}MHz)
// Generates: {covered_targets} MHz
NX_PLL_U #(
    .location        (""),
    .ref_osc_on      ({ref_osc}),
    .use_pll         (1'b1),
    .ext_fbk_on      (1'b0),
    .fbk_delay_on    (1'b0),
    .fbk_delay       (6'd0),
    // ref_intdiv register = divide ratio (hardware-validated golden: 375/15 = 25 MHz PFD);
    // cfg['ref'] stores ratio-1 internally, so emit +1.
    .ref_intdiv      (5'd{cfg['ref'] + 1}),
    .fbk_intdiv      (7'd{cfg['fbk']}),
    .clk_outdiv1     (3'd{g['clk_outdiv1'][0]}),
    .clk_outdiv2     (3'd{g['clk_outdiv2'][0]}),
    .clk_outdiv3     (3'd{g['clk_outdiv3'][0]}),
    .clk_outdiv4     (3'd{g['clk_outdiv4'][0]}),
    .clk_outdivd1    (4'd{g['clk_outdivd1'][0]}),
    .clk_outdivd2    (4'd{g['clk_outdivd2'][0]}),
    .clk_outdivd3    (4'd{g['clk_outdivd3'][0]}),
    .clk_outdivd4    (4'd{g['clk_outdivd4'][0]}),
    .clk_outdivd5    (4'd{g['clk_outdivd5'][0]})
) PLL_{pll_counter} (
    .REF       ({ref_conn}),
    .FBK       (1'b0),
    .R         (rst), // active high reset
    .VCO       (),
    .LDFO      (),
    .REFO      (),
    .OSC       (),
    .CAL_LOCKED(),
    .PLL_LOCKED(locked_{pll_counter}),
    .CLK_DIV1  {ports_map['CLK_DIV1']},
    .CLK_DIV2  {ports_map['CLK_DIV2']},
    .CLK_DIV3  {ports_map['CLK_DIV3']},
    .CLK_DIV4  {ports_map['CLK_DIV4']},
    .CLK_DIVD1 {ports_map['CLK_DIVD1']},
    .CLK_DIVD2 {ports_map['CLK_DIVD2']},
    .CLK_DIVD3 {ports_map['CLK_DIVD3']},
    .CLK_DIVD4 {ports_map['CLK_DIVD4']},
    .CLK_DIVD5 {ports_map['CLK_DIVD5']}
);"""
        pll_counter += 1

    return verilog
