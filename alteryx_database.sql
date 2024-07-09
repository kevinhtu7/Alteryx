CREATE TABLE Users (
    UserID VARCHAR(255) PRIMARY KEY,
    PW VARCHAR(255),
    role VARCHAR(50),
    FOREIGN KEY (role) REFERENCES Roles(role)
);

DROP TABLE Users;
CREATE TABLE Roles (
    role VARCHAR(50) PRIMARY KEY,
    access_levels VARCHAR(255)
);

-- Insert data into Roles table
INSERT INTO Roles (role, access_levels) VALUES ('CEO', 'Executive');
INSERT INTO Roles (role, access_levels) VALUES ('CFO', 'Executive');
INSERT INTO Roles (role, access_levels) VALUES ('Analyst', 'General');
INSERT INTO Roles (role, access_levels) VALUES ('Engineer', 'General');
INSERT INTO Roles (role, access_levels) VALUES ('Associate', 'General');
INSERT INTO Roles (role, access_levels) VALUES ('Consultant', 'General');

-- Insert data into Users table
INSERT INTO Users (UserID, PW, role) VALUES ('seal_kala', 'DB4BSANcourses', 'CEO');
INSERT INTO Users (UserID, PW, role) VALUES ('seyyed_salili', 'DB4BSANcourses', 'CFO');
INSERT INTO Users (UserID, PW, role) VALUES ('aaron_sornborger', 'DB4BSANcourses', 'Analyst');
INSERT INTO Users (UserID, PW, role) VALUES ('mason_dosher', 'DB4BSANcourses', 'Engineer');
INSERT INTO Users (UserID, PW, role) VALUES ('herat_devisha', 'DB4BSANcourses', 'Associate');
INSERT INTO Users (UserID, PW, role) VALUES ('kevin_tu', 'DB4BSANcourses', 'Consultant');

-- Display data from Users table
SELECT * FROM Users;

-- Display data from Roles table
 SELECT * FROM Roles;

SELECT access_levels
FROM Users
JOIN Roles
ON Users.role = Roles.role
WHERE UserID = 'kevin_tu';

