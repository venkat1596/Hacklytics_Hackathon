"use client";

export default function NotebookViewer() {
  const gistUrl = "https://gist.github.com/Adonalsiun/5897c3c2e832840101a4b8026eff5965";

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      

      {/* Main Content Area */}
      <div style={{ flex: 1, position: 'relative' }}>
        {/* IFrame taking up the remaining space */}
        <iframe
          src={`${gistUrl}.pibb`}
          width="100%"
          height="100%"
          style={{
            border: 'none',
            position: 'absolute', // Position the iframe absolutely within the container
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
          }}
        />
      </div>
    </div>
  );
}