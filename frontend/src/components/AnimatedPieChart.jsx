// src/components/AnimatedPieChart.jsx
import React, { useState, useCallback } from 'react';
import Pie from '@visx/shape/lib/shapes/Pie';
import { scaleOrdinal } from '@visx/scale';
import { Group } from '@visx/group';
import { animated, useTransition, interpolate } from '@react-spring/web';
import { localPoint } from '@visx/event';
import { Tooltip } from '@visx/tooltip';

// Outer ring colors
const getStockColor = scaleOrdinal({
  range: [
    'rgba(255, 99, 132, 0.8)',
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(255, 120, 180, 0.8)', // Pinkish tone
    'rgba(0, 128, 128, 0.8)',   // Teal tone
    'rgba(128, 0, 128, 0.8)',   // Purple tone
    'rgba(255, 165, 0, 0.8)',   // Orange tone
    'rgba(0, 100, 0, 0.8)',     // Dark green tone
    'rgba(128, 128, 0, 0.8)' ,  // Olive tone
    'rgba(255, 20, 147, 0.8)',  // Deep pink tone
    'rgba(70, 130, 180, 0.8)',  // Steel blue tone
    'rgba(255, 140, 0, 0.8)',   // Dark orange tone
    'rgba(34, 139, 34, 0.8)',   // Forest green tone
    'rgba(106, 90, 205, 0.8)',  // Slate blue tone
    'rgba(220, 20, 60, 0.8)'    // Crimson tone


  ]
});

// Inner ring colors - darker shades
const getInnerRingColor = scaleOrdinal({
  range: [
    'rgba(255, 129, 162, 0.8)',  // Brighter than rgb(255, 99, 132)
    'rgba(104, 202, 255, 0.8)',  // Brighter than rgb(54, 162, 235)
    'rgba(255, 226, 106, 0.8)',  // Brighter than rgb(255, 206, 86)
    'rgba(125, 222, 222, 0.8)',  // Brighter than rgb(75, 192, 192)
    'rgba(203, 152, 255, 0.8)',  // Brighter than rgb(153, 102, 255)
    'rgba(255, 189, 94, 0.8)',   // Brighter than rgb(255, 159, 64)
    'rgba(255, 150, 210, 0.8)',  // Brighter than rgb(255, 120, 180)
    'rgba(50, 178, 178, 0.8)',   // Brighter than rgb(0, 128, 128)
    'rgba(178, 50, 178, 0.8)',   // Brighter than rgb(128, 0, 128)
    'rgba(255, 195, 50, 0.8)',   // Brighter than rgb(255, 165, 0)
    'rgba(50, 150, 50, 0.8)',    // Brighter than rgb(0, 100, 0)
    'rgba(178, 178, 50, 0.8)',   // Brighter than rgb(128, 128, 0)
    'rgba(255, 50, 177, 0.8)',   // Brighter than rgb(255, 20, 147)
    'rgba(120, 180, 230, 0.8)',  // Brighter than rgb(70, 130, 180)
    'rgba(255, 160, 50, 0.8)',   // Brighter than rgb(255, 140, 0)
    'rgba(84, 189, 84, 0.8)',    // Brighter than rgb(34, 139, 34)
    'rgba(156, 140, 255, 0.8)',  // Brighter than rgb(106, 90, 205)
    'rgba(240, 50, 90, 0.8)'     // Brighter than rgb(220, 20, 60)

  ]
});

const defaultMargin = { top: 20, right: 20, bottom: 20, left: 20 };

const fromLeaveTransition = ({ endAngle }) => ({
  startAngle: endAngle > Math.PI ? 2 * Math.PI : 0,
  endAngle: endAngle > Math.PI ? 2 * Math.PI : 0,
  opacity: 0,
});

const enterUpdateTransition = ({ startAngle, endAngle }) => ({
  startAngle,
  endAngle,
  opacity: 1,
});

// Tooltip component
const TooltipContent = ({ stock }) => (
    <div
      style={{
        backgroundColor: 'rgba(27, 32, 40, 0.9)',
        padding: '12px',
        borderRadius: '6px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        color: 'white',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <img
          src={`https://logo.clearbit.com/${stock.security?.toLowerCase().replace(/[^a-z0-9]/g, '')}.com`}
          alt={stock.symbol}
          style={{
            width: '24px',
            height: '24px',
            borderRadius: '50%',
            backgroundColor: 'white',
            padding: '2px'
          }}
          onError={(e) => e.target.src = '/default-stock-logo.png'}
        />
        <strong>{stock.symbol}</strong> - {stock.security}
      </div>
      <div>Weight: {(stock.weight * 100).toFixed(2)}%</div>
    </div>
  );

function AnimatedPie({ animate, arcs, path, getKey, getColor, onClickDatum, getTooltipData }) {
  const [tooltipData, setTooltipData] = useState(null);
  const [tooltipLeft, setTooltipLeft] = useState(0);
  const [tooltipTop, setTooltipTop] = useState(0);

  const handleMouseMove = useCallback(
    (event, arc) => {
      const { clientX, clientY } = event;
      setTooltipData(getTooltipData(arc));
      setTooltipLeft(clientX);
      setTooltipTop(clientY);
    },
    [getTooltipData]
  );

  const handleMouseLeave = () => {
    setTooltipData(null);
  };

  const transitions = useTransition(arcs, {
    from: animate ? fromLeaveTransition : enterUpdateTransition,
    enter: enterUpdateTransition,
    update: enterUpdateTransition,
    leave: animate ? fromLeaveTransition : enterUpdateTransition,
    keys: getKey,
  });

  return (
    <>
      {transitions((props, arc, { key }) => {
        const [centroidX, centroidY] = path.centroid(arc);
        const hasSpaceForLabel = arc.endAngle - arc.startAngle >= 0.1;

        return (
          <g key={key}>
            <animated.path
              d={interpolate([props.startAngle, props.endAngle], (startAngle, endAngle) =>
                path({
                  ...arc,
                  startAngle,
                  endAngle,
                }),
              )}
              fill={getColor(arc)}
              onMouseMove={(e) => handleMouseMove(e, arc)}
              onMouseLeave={handleMouseLeave}
              onClick={() => onClickDatum(arc)}
              onTouchStart={() => onClickDatum(arc)}
              style={{ cursor: 'pointer' }}
            />
            {hasSpaceForLabel && (
              <animated.g style={{ opacity: props.opacity }}>
                <text
                  fill="white"
                  x={centroidX}
                  y={centroidY}
                  dy=".33em"
                  fontSize={12}
                  textAnchor="middle"
                  pointerEvents="none"
                >
                  {getKey(arc)}
                </text>
              </animated.g>
            )}
          </g>
        );
      })}
      {tooltipData && (
        <Tooltip
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            position: 'fixed',
            transform: 'translate(-50%, -100%)',
            pointerEvents: 'none',
          }}
        >
          <TooltipContent stock={tooltipData} />
        </Tooltip>
      )}
    </>
  );
}

const PortfolioPieChart = ({ width, height, data, margin = defaultMargin, animate = true }) => {
    const [selectedStock, setSelectedStock] = useState(null); // Add state for selectedStock

    if (width < 10 || height < 10) return null;

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const radius = Math.min(innerWidth, innerHeight) / 2;
    const centerY = innerHeight / 2;
    const centerX = innerWidth / 2;
    const gap = 10; // Adjust gap size here

    const outerRingData = data.map(item => ({
      ...item,
      value: item.weight * 100
    }));

    const innerCircleData = data.map(item => ({
      ...item,
      value: item.weight * 100
    }));

    return (
      <svg width={width} height={height}>
        <Group top={centerY + margin.top} left={centerX + margin.left}>
          <Pie
            data={selectedStock ? outerRingData.filter(d => d.symbol === selectedStock) : outerRingData}
            pieValue={d => d.value}
            outerRadius={(radius - gap)* 1.1}
            innerRadius={(radius * 0.67 + gap)*1.1}
            cornerRadius={3}
            padAngle={0.005}
          >
            {pie => (
              <AnimatedPie
                {...pie}
                animate={animate}
                getKey={arc => arc.data.symbol}
                onClickDatum={({ data }) =>
                  animate && setSelectedStock(selectedStock && selectedStock === data.symbol ? null : data.symbol)
                }
                getColor={arc => getStockColor(arc.data.symbol)}
                getTooltipData={arc => arc.data}
              />
            )}
          </Pie>
          <Pie
            data={selectedStock ? innerCircleData.filter(d => d.symbol === selectedStock) : innerCircleData}
            pieValue={d => d.value}
            outerRadius={(radius * 0.6)*1.2}
            innerRadius={0}
            cornerRadius={3}
            padAngle={0.005}
          >
            {pie => (
            <AnimatedPie
                {...pie}
                animate={animate}
                getKey={arc => `${(arc.data.weight * 100).toFixed(2)}%`}  // แสดงเปอร์เซ็นต์น้ำหนักแทนชื่อหุ้น
                onClickDatum={({ data }) =>
                animate && setSelectedStock(selectedStock && selectedStock === data.symbol ? null : data.symbol)
                }
                getColor={arc => getInnerRingColor(arc.data.symbol)}
                getTooltipData={arc => arc.data}
            />
            )}

          </Pie>
        </Group>
      </svg>
    );
};

export default PortfolioPieChart;