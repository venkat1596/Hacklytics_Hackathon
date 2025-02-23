import { NavbarSimpleColored } from "../lib/SideBar"; // Sidebar/Navbar
import styles from "../page.module.css"; // Import styles
import { Banner } from "../lib/Banner";
import DrawIOEmbed from "../lib/Workflow";

export default function WorkflowPage() {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored /> {/* Sidebar */}
      <div className={styles.content}>
        <main>
          <Banner />
          <DrawIOEmbed
            diagramUrl="https://drive.google.com/uc?id=1M-iW6nt1wAC8jlOhwZTF4CkuQu937ie8&export=download"
            width="100%"
            height="800px"
          />
        </main>
      </div>
    </div>
  );
}