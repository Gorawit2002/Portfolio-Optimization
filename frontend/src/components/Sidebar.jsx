import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <ul className="menu">
        <li>
          <NavLink 
            to="/" 
            
            className={({ isActive }) => (isActive ? 'active' : '')}
          >
            Overview
          </NavLink>
        </li>
        <li>
          <NavLink 
            to="/optimize" 
            className={({ isActive }) => isActive ? 'active optimize-btn' : 'optimize-btn'}
          >
            Optimize
          </NavLink>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;
