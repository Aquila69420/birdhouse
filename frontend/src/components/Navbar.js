import React from "react";
import { Link } from "react-router-dom";
import styles from "../styles/Navbar.module.css"; // Import CSS module

const Navbar = () => {
  return (
    <nav className={styles.navbar}>
      <div className={styles.navbarBrand}>
        <Link to="/" className={styles.navbarLogo}>
          Bird House
        </Link>
      </div>
      <ul className={styles.navbarLinks}>
        <li className={styles.navbarLinkItem}><Link to="/login" className={styles.navbarLink}>Login</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/signup" className={styles.navbarLink}>Signup</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/build" className={styles.navbarLink}>Build</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/train" className={styles.navbarLink}>Train</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/validate" className={styles.navbarLink}>Validate</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/deploy" className={styles.navbarLink}>Deploy</Link></li>
        <li className={styles.navbarLinkItem}><Link to="/data" className={styles.navbarLink}>Data</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
