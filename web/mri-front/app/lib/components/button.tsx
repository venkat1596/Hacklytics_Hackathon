import { FunctionComponent } from "react";

export type ButtonType = {
  className?: string;
  button?: string;

  /** Variant props */
  darkMode?: boolean;
  iconPosition?: string;
  small?: boolean;
  style?: "Primary" | "Secondary";
};

const Button: FunctionComponent<ButtonType> = ({
  className = "",
  darkMode = false,
  iconPosition = "No icon",
  small = false,
  style = "Primary",
  button,
}) => {
  return (
    <button
      style={{
        cursor: "pointer",
        border: "1px solid #000",
        padding: "12px 24px",
        backgroundColor: "#000",
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
      }}
      className={className}
      data-small={small}
      data-style={style}
    >
      <div
        style={{
          position: "relative",
          fontSize: "16px",
          lineHeight: "150%",
          fontFamily: "Roboto",
          color: "#fff",
          textAlign: "left",
        }}
      >
        {button}
      </div>
    </button>
  );
};

export default Button;
