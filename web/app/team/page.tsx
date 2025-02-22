import React from 'react';
import { HeaderSimple } from '../lib/HeaderSimple';
import { FooterCentered } from '../lib/FooterCentered';
import { FeaturesCards } from '../lib/FeaturesCards';
import { NavbarSimpleColored } from '../lib/SideBar'; // Import the navbar
import styles from '../page.module.css'; // Import the same styles

const TeamPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored /> {/* Add the navbar */}
      <div className={styles.content}>
        <HeaderSimple />
        <main>
          <FeaturesCards />
        </main>
      </div>
    </div>
  );
};

export default TeamPage;