"""
Tee bending resistance at the centre-line of a circular web opening.

This module provides the Tee-section resistances used by the Enhanced-RSM
plastic-capacity analysis, in particular the reduced plastic bending resistance
under combined high shear and global bending, which serves as one of the two
stop criteria for moment redistribution.

Kept in its own module so that the analysis-mode scripts share a single
definition rather than repeating it.

Units follow the package convention: lengths in mm, yield strength in MPa,
forces in kN and moments in kNm at the interfaces of this module.
"""


class VierendeelCircularOpening:
    """Tee-section resistances at the centre-line of a circular web opening.

    Provides the plastic and elastic bending resistances, the shear resistance
    and the reduced plastic bending resistance under combined high shear and
    global bending. The reduced resistance is used as one of the two stop
    criteria for moment redistribution in the plastic-capacity analysis.
    """

    def __init__(self, d_o, h, b_f, t_f, t_w, r, f_y):
        """Initialise the opening.

        Parameters
        ----------
        d_o : float
            Opening diameter (mm).
        h : float
            Section depth (mm).
        b_f : float
            Flange width (mm).
        t_f : float
            Flange thickness (mm).
        t_w : float
            Web thickness (mm).
        r : float
            Root radius (mm).
        f_y : float
            Yield strength (MPa).
        """
        self.d_o = d_o
        self.h = h
        self.b_f = b_f
        self.t_f = t_f
        self.t_w = t_w
        self.r = r
        self.f_y = f_y
        self.h_T = (h - d_o) / 2
        self.d_T = self.h_T - t_f

    def plastic_bending_resistance(self, t_w_eff=None):
        """Plastic bending resistance of the Tee (kNm) and its plastic n.a. depth."""
        if t_w_eff is None:
            t_w_eff = self.t_w
        z_pl = (t_w_eff * self.d_T + self.b_f * self.t_f) / (2 * self.b_f)
        term1 = t_w_eff * self.d_T * (self.d_T / 2 + (self.t_f - z_pl))
        term2 = self.b_f * (z_pl**2 + (self.t_f - z_pl)**2) / 2
        M_pl_T = (term1 + term2) * self.f_y * 1e-6
        return M_pl_T, z_pl

    def elastic_bending_resistance(self):
        """Elastic bending resistance of the Tee (kNm), elastic n.a. depth and I."""
        numerator = (
            self.b_f * (self.t_f**2 / 2)
            + self.t_w * self.d_T * (self.t_f + self.d_T / 2)
        )
        denominator = self.t_w * self.d_T + self.b_f * self.t_f
        z_el = numerator / denominator
        I_T = (
            self.b_f * self.t_f * (z_el - self.t_f / 2)**2
            + self.t_w * self.d_T * (self.t_f + self.d_T / 2 - z_el)**2
            + (self.b_f * self.t_f**3) / 12
            + (self.t_w * self.d_T**3) / 12
        )
        M_el_T = I_T * self.f_y / (self.h_T - z_el) * 1e-6
        return M_el_T, z_el, I_T

    def shear_resistance(self, t_w_eff=None):
        """Tee and total pure-shear resistance at the centre-line (kN)."""
        if t_w_eff is None:
            t_w_eff = self.t_w
        V_T_Rd = (
            t_w_eff * self.d_T + 0.5 * self.t_f * (2 * self.r + t_w_eff)
        ) * 0.577 * self.f_y * 1e-3
        V_Rd = 2 * V_T_Rd
        return V_T_Rd, V_Rd

    def effective_web_thickness(self, V_ratio):
        """Effective web thickness under high shear, per SCI P355."""
        return (1 - (2 * V_ratio - 1)**2) * self.t_w

    def global_bending_interaction(self, V_Ed, moment_shear_ratio):
        """Reduced Tee plastic bending resistance under shear and global bending.

        Applies the SCI P355 web-thickness reduction when the shear ratio
        exceeds 0.5 and the moment-axial interaction reduction from global
        bending, returning the reduced Tee plastic bending resistance (kNm).

        Parameters
        ----------
        V_Ed : float
            Applied shear at the centre-line (kN).
        moment_shear_ratio : float
            Global moment-to-shear ratio at the centre-line.

        Returns
        -------
        float
            Reduced Tee plastic bending resistance ``M_T_Rd`` (kNm).
        """
        A_T = self.t_w * self.d_T + self.b_f * self.t_f
        _, z_el, _ = self.elastic_bending_resistance()
        h_eff = self.h - 2 * z_el

        # Bending resistance of the beam at the centre-line of the opening.
        M_Rd = A_T * h_eff * self.f_y * 1e-6

        # Web-thickness reduction under high shear.
        _, V_Rd = self.shear_resistance()
        M_Ed = V_Ed * moment_shear_ratio
        N_T_Ed = M_Ed * 1e3 / h_eff  # kN
        V_ratio = V_Ed / V_Rd

        if V_ratio > 0.5:
            t_w_eff = self.effective_web_thickness(V_ratio)
            A_T_eff = t_w_eff * self.d_T + self.b_f * self.t_f
            N_pl_T = A_T_eff * self.f_y * 1e-3  # kN
        else:
            t_w_eff = self.t_w
            N_pl_T = A_T * self.f_y * 1e-3  # kN

        # Vierendeel resistance with the current web thickness, reduced for the
        # global-bending axial force.
        M_pl_T, _ = self.plastic_bending_resistance(t_w_eff)
        N_ratio = N_T_Ed / N_pl_T
        interaction_factor = 1 - N_ratio**2
        M_T_Rd = M_pl_T * interaction_factor
        return M_T_Rd


def analyze_circular_opening(d_o, h, b_f, t_f, t_w, r, f_y,
                             moment_shear_ratio=None, V_Ed=None):
    """Reduced Tee plastic bending resistance for a circular opening.

    Thin convenience wrapper around :class:`VierendeelCircularOpening` used by
    the plastic-capacity analysis to obtain the redistribution stop criterion.

    Parameters
    ----------
    d_o, h, b_f, t_f, t_w, r, f_y : float
        Opening diameter, section depth, flange width, flange thickness, web
        thickness, root radius (mm) and yield strength (MPa).
    moment_shear_ratio : float, optional
        Global moment-to-shear ratio at the centre-line. If omitted, no
        interaction reduction is computed and the function returns ``None``.
    V_Ed : float, optional
        Applied shear at the centre-line (kN).

    Returns
    -------
    float or None
        Reduced Tee plastic bending resistance ``M_T_Rd`` (kNm), or ``None`` if
        ``moment_shear_ratio`` is not supplied.
    """
    opening = VierendeelCircularOpening(d_o, h, b_f, t_f, t_w, r, f_y)
    if moment_shear_ratio:
        return opening.global_bending_interaction(V_Ed, moment_shear_ratio)
    return None
