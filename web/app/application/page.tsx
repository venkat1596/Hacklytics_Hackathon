import React from "react";
import { NavbarSimpleColored } from "../lib/SideBar"; // Sidebar/Navbar
import NotebookViewer from "../lib/ApplicationShell"; // Import the Notebook Viewer
import styles from "../page.module.css"; // Import styles
import { HeroText } from "../lib/HeroText"; // Import HeroText
import { DropzoneButton } from "../lib/DropzoneButton"; // Import DropzoneButton

const ApplicationPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored /> {/* Sidebar */}
      <div className={styles.content}>
        <main>
          <HeroText />
          <DropzoneButton /> {/* Use DropzoneButton instead of Dropzone */}
        </main>
      </div>
    </div>
  );
};

export default ApplicationPage;

