/** Soft aurora background blob, layered behind hero content. Pure
 * CSS — see `.aurora` in globals.css for the radial-gradient + drift
 * animation. Wrapped as a component so layout files don't carry
 * presentational divs. */
export function Aurora() {
    return <div className="aurora" aria-hidden />;
}
