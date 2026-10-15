import React, { useState } from "react";
import { Button, Container, Box } from "@mui/material";

function TwoButtons({ onClick }) {
//   const [selectedCondition, setSelectedCondition] = useState("");

  const handleLeftClick = () => {
    // const updatedValue = 1;
    // setSelectedCondition(updatedValue);
    onClick(1);
  };

  const handleRightClick = () => {
    // const updatedValue = 2;
    // setSelectedCondition(updatedValue);
    onClick(2);
  };


  return (
    <Container>
      <Box
        display="flex"
        flexDirection="row"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        sx={{ gap: 2 }}
      >
        <Button
          variant="contained"
          color={"primary"}
          onClick={handleLeftClick}
        >
          Select condition 1
        </Button>
        <Button
          variant="contained"
          color={"primary"}
          onClick={handleRightClick}
        >
          Select condition 2
        </Button>
      </Box>
    </Container>
  );
}

export default TwoButtons;