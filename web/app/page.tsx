import { HeaderSimple } from "./lib/HeaderSimple";
import { HeroBullets } from "./lib/HeroBullets";
import { FooterCentered } from "./lib/FooterCentered";
import { NavbarSimpleColored } from "./lib/SideBar";
import styles from "./page.module.css"; // Ensure styles are properly defined

export default function Home() {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored />
      <div className={styles.content}>
        <HeaderSimple />
        <HeroBullets />
      </div>
    </div>
  );
}
