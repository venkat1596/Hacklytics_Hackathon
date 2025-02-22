import React from "react";
import { NavbarSimpleColored } from "../lib/SideBar"; // Sidebar/Navbar
import NotebookViewer from "../lib/ApplicationShell"; // Import the Notebook Viewer
import styles from "../page.module.css"; // Import the same styles
import { HeroText } from "../lib/HeroText"; // Import the HeroText component

const ApplicationPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored /> {/* Sidebar */}
      <div className={styles.content}>
        <main>
        <HeroText />
          <NotebookViewer /> {/* Embed Notebook Viewer */}
        </main>
      </div>
    </div>
  );
};

export default ApplicationPage;
