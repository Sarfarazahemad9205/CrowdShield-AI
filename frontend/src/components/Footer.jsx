import { Link } from "react-router-dom";
import { Brand } from "./Header";
export default function Footer(){return <footer className="site-footer"><Brand/><span>Thoughtful technology for safer spaces.</span><nav aria-label="Footer navigation"><Link to="/benefits">Benefits</Link><Link to="/about">About</Link><Link to="/faqs">FAQs</Link></nav><span className="copyright">© {new Date().getFullYear()} StampedeGuard</span></footer>}
