import meep as mp  # type: ignore


def add_flux_box(
    size: mp.Vector3,
    center: mp.Vector3 = mp.Vector3(),
) -> list[mp.FluxRegion]:
    """
    Adds 6 flux regions to create a box around the center point with the given
    size. Specifies the weight parameter on each assuming we care about flux
    leaving the box (and not entering)
    """

    out: list[mp.FluxRegion] = []

    top = mp.FluxRegion(
        center=center + mp.Vector3(0, 0, size.z / 2),
        size=mp.Vector3(size.x, size.y, 0),
    )

    bot = mp.FluxRegion(
        center=center - mp.Vector3(0, 0, size.z / 2),
        size=mp.Vector3(size.x, size.y, 0),
        weight=-1,
    )

    x_r = mp.FluxRegion(
        center=center + mp.Vector3(size.x / 2, 0, 0),
        size=mp.Vector3(0, size.y, size.z),
    )

    x_l = mp.FluxRegion(
        center=center + mp.Vector3(-size.x / 2, 0, 0),
        size=mp.Vector3(0, size.y, size.z),
        weight=-1,
    )

    y_b = mp.FluxRegion(
        center=center + mp.Vector3(0, size.y / 2, 0),
        size=mp.Vector3(size.x, 0, size.z),
    )

    y_f = mp.FluxRegion(
        center=center + mp.Vector3(0, -size.y / 2, 0),
        size=mp.Vector3(size.x, 0, size.z),
        weight=-1,
    )

    out.extend([top, bot, x_r, x_l, y_b, y_f])

    return out


def get_total_box_flux(fluxes: list[mp.DftFlux]) -> float:
    flux: float = 0

    for f in fluxes:
        temp = mp.get_fluxes(f)  # type: ignore
        flux += temp[0]  # TODO: check!

    return flux
