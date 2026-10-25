import { useEffect, useState } from "react";
import { ArrowUpRight, Menu, ShieldCheck, X } from "lucide-react";
import { Link, NavLink } from "react-router-dom";

const links = [["Benefits", "/benefits"], ["How it works", "/how-it-works"], ["About", "/about"], ["FAQs", "/faqs"]];
export function Brand() { return <Link className="brand" to="/" aria-label="StampedeGuard home"><span className="brand-mark"><ShieldCheck size={19}/></span><span>Stampede<span className="brand-muted">Guard</span></span></Link>; }
export default function Header() {
  const [open, setOpen] = useState(false);
  useEffect(() => { const esc = e => e.key === "Escape" && setOpen(false); const resize = () => window.innerWidth > 760 && setOpen(false); window.addEventListener("keydown", esc); window.addEventListener("resize", resize); document.body.classList.toggle("menu-open", open); return () => { window.removeEventListener("keydown", esc); window.removeEventListener("resize", resize); document.body.classList.remove("menu-open"); }; }, [open]);
  return <><header className="site-header"><Brand/><nav className={`site-nav${open ? " is-open" : ""}`} aria-label="Primary navigation">{links.map(([name, path]) => <NavLink end key={path} to={path} onClick={() => setOpen(false)}>{name}</NavLink>)}</nav><Link className="button button-light header-cta" to="/video-analysis">Analyze video <ArrowUpRight size={15}/></Link><button className="menu-toggle" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen(!open)}>{open ? <X size={21}/> : <Menu size={21}/>}</button></header>{open && <button className="menu-backdrop visible" aria-label="Close navigation" onClick={() => setOpen(false)}/>}</>;
}
