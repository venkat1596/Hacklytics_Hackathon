import React from 'react';

interface JupyterNotebookViewerProps {
  notebookUrl: string; // Define the notebookUrl prop
}

const JupyterNotebookViewer: React.FC<JupyterNotebookViewerProps> = ({ notebookUrl }) => {
  return (
    <div style={{ width: '100%', height: '100vh', overflow: 'hidden' }}>
      <iframe
        src={notebookUrl}
        style={{ width: '100%', height: '100%', border: 'none' }}
        title="Jupyter Notebook Viewer"
      />
    </div>
  );
};

export default JupyterNotebookViewer;