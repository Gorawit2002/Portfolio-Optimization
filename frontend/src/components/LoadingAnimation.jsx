import React from 'react';
import FadeIn from "react-fade-in";
import Lottie from "react-lottie";
import * as loadingData from "../animations/loading.json";
import * as successData from "../animations/success.json";

const LoadingAnimation = ({ loading, success }) => {
  const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: loadingData.default,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice"
    }
  };

  const successOptions = {
    loop: true,
    autoplay: true,
    animationData: successData.default,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice"
    }
  };

  return (
    <div className="loading-animation-container">
      <FadeIn>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <h1 style={{ color: "white", marginBottom: "20px" }}>
            {!success ? "Optimizing Portfolio..." : "Optimization Complete"}
          </h1>
          <Lottie
            options={!success ? defaultOptions : successOptions}
            height={140}
            width={140}
          />
        </div>
      </FadeIn>
    </div>
  );
};

export default LoadingAnimation;