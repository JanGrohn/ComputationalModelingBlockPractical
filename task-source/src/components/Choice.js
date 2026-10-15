import React, { useState, useEffect, useRef } from "react";
import {
    Box,
    Paper,
    Button,
    Typography,
    Slide,
    Stack,
    Slider,
    Grow,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    Alert,
    AlertTitle,
    DialogActions,
    IconButton,
    Collapse,
} from "@mui/material";
import Image1 from "./img/bird-crane-shape.svg";
import Image2 from "./img/deer-silhouette.svg";
// import Image3 from "./img/horse-shape.svg";
// import Image4 from "./img/kangaroo-shape.svg";
import TextTransition, { presets } from "react-text-transition";
import CheckIcon from "@mui/icons-material/Check";
import ForwardIcon from '@mui/icons-material/Forward';
import { styled } from "@mui/system";
// import { saveData } from "./save";
import CloseIcon from '@mui/icons-material/Close';
import EndStudy from "./EndStudy";

const PaperWrapper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    backgroundColor: "white",
    borderRadius: "12px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    flexGrow: 1,
    flexBasis: 0,
    maxHeight: "50vh",
    maxWidth: "50vw",
}));

function shuffle(array) {
    let currentIndex = array.length, randomIndex;

    while (currentIndex != 0) {

        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }

    return array;
}


var reward1 = [];
var reward2 = [];


for (var i = 0; i < 100; i++) {
  reward1.push(Math.floor(Math.random() * 99) + 1);
  reward2.push(Math.floor(Math.random() * 99) + 1);

}
  

let zeroes = new Array(reward1.length).fill(0);
let ones = new Array(reward1.length).fill(1);
let displaySide = shuffle([...zeroes, ...ones]);



reward1.unshift("66", "45")
reward2.unshift("31", "60")
displaySide.unshift(0, 1)



function Choice(props) {
    

    // let oneRewarded = [];

    let prob1 = new Array(103).fill(0.25);
    let prob2_1 = new Array(28).fill(0.25);
    let prob2_2 = new Array(25).fill(0.75);
    let prob2_3 = new Array(25).fill(0.25);
    let prob2_4 = new Array(25).fill(0.75);
    let prob2 = [...prob2_1, ...prob2_2, ...prob2_3, ...prob2_4];

    if (props.condition==1) {
        var prob = prob1;
    } else if (props.condition==2) {
        var prob = prob2;
    }

    // for (var i = 0; i < 100; i++) {
    //     if (Math.random() > prob[i]) {
    //         oneRewarded.push(true);
    //     } else {
    //         oneRewarded.push(false);
    //     }
    // }

    const [oneRewarded, setOneRewarded] = useState(true);
    // useEffect(() => {
    //     console.log(prob[currentIndex])
    //     if (Math.random() > prob[currentIndex]) {
    //         setOneRewarded(true);
    //     } else {
    //         setOneRewarded(false);
    //     }

    //   }, [oneRewarded]);


    useEffect(() => {
        const interval = setInterval(() => {
            // handleSubmit(window.participantData)
        }, 60000);

        // Clean up: clear the interval when the component is unmounted
        return () => {
            clearInterval(interval);
        };
    }, []);

    const [selectedPaper, setSelectedPaper] = useState(null);
    const [confirmDisabled, setConfirmDisabled] = useState(true);
    const [newPaperAnimation, setNewPaperAnimation] = useState(true);
    const containerRef = useRef(null);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [sliderValue, setSliderValue] = useState(5);
    const [sliderMoved, setSliderMoved] = useState(false);
    const [showSlider, setShowSlider] = useState(false);

    const handlePaperClick = (paper) => { 
            window.participantData.push({
            trial: currentIndex,
            time: new Date().getTime(),
            type: paper === "left" ? "Select Left" : "Select Right",
            displaySide: displaySide[currentIndex],
            reward1: reward1[currentIndex],
            reward2: reward2[currentIndex],
            oneRewarded: oneRewarded,
            });
      
        setSelectedPaper(paper);
          setTimeout(() => {
            setConfirmDisabled(false);
            setSelectedPaper(paper);
          }, 1000);
      };


    const handleSliderConfirm = () => {
        window.participantData.push({
            trial: currentIndex,
            time: new Date().getTime(),
            type: "Confirm Slider",
            sliderValue: sliderValue,
        });
        setShowSlider(false);
        setTimeout(() => {
            setConfirmDisabled(false);
        }, 2000);
        setSliderMoved(false);
    };


    useEffect(() => {
        if (showSlider) {
            setSliderValue(5);
        }
    }, [showSlider]);

    const handleConfirmClick = () => {

        window.participantData.push({ trial: currentIndex + 1, time: new Date().getTime(), type: "Trial started" });
        
        setConfirmDisabled(true);
        setNewPaperAnimation(false);
        setTimeout(() => {
            setSelectedPaper(null);
            setNewPaperAnimation(true);
            setCurrentIndex(currentIndex + 1);
            if (Math.random() > prob[currentIndex]) {
                setOneRewarded(true);
            } else {
                setOneRewarded(false);
            }
        }, 1000);
    };

    if (currentIndex >= reward1.length) {
        return (
            <EndStudy url="https://app.prolific.co/submissions/complete?cc=CNEI59LT" />
        );
    }

    

    return (
        <Box
            sx={{
                backgroundColor: "white",
                minHeight: "100vh",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                width: { xs: 5 / 10, sm: 5 / 10 },
                position: "absolute",
                left: "50%",
                top: "50%",
                transform: "translate(-50%, -50%)",
            }}
        >

            <Box
                style={{
                    overflow: "hidden"
                }}
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    minHeight: "10vh",
                    backgroundColor: "#f5f5f5",
                    borderRadius: "8px",
                    boxShadow: "0px 0px 16px rgba(0, 0, 0, 0.12)",
                    padding: "2rem",
                }}
                ref={containerRef}
            >
                <Collapse in={Boolean(currentIndex === 0)}><Alert icon={false} severity="info" sx={{ fontSize: "h7", mb: 2 }}

                ><AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                    <strong><>You are repeatedly choosing between two pictures. You have to select which picture you want to choose by clicking the corresponding button. The point amount you can win from selecting a picture is shown below it. Choose one of the pictures now.</></strong>
                </Alert></Collapse>
                <Collapse in={Boolean(currentIndex === 1)}><Alert icon={false} severity="warning" sx={{ fontSize: "h7", mb: 2 }}

                ><AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                    <strong><>As you saw, only one of the pictures actually yields points after your choice. The overall points you won from the pictures you choose directly translate into a bonus payment at the end of the study. It is your task to figure out which picture will be rewarded next. Choose one of the pictures again now.  </></strong>
                </Alert></Collapse>
                <Collapse in={Boolean(currentIndex === 2)}><Alert icon={false} severity="success" sx={{ fontSize: "h7", mb: 2 }}

                ><AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                    <strong><> Try to get as many points as possible to receive a larger bonus payment. Continue choosing between the pictures. The study starts now.</></strong>
                </Alert></Collapse>


                {(<Stack
                    direction="row"
                    justifyContent="space-around"
                    width="100%"
                    spacing={4}
                >
                    <Slide direction="right" in={newPaperAnimation} container={containerRef.current}>
                        <PaperWrapper
                            elevation={3}
                            sx={{
                                backgroundColor:
                                    selectedPaper === "left" ? "lightblue" : "white",
                            }}
                        >
                            <div style={{ userSelect: "none", display: "flex", justifyContent: "center", alignItems: "center", height: "100%", flexGrow: 1, overflow: "hidden" }}>
                                <img
                                    src={displaySide[currentIndex] ? Image1 : Image2}
                                    displaySide
                                    style={{
                                        maxWidth: "100%",
                                        maxHeight: "100%",
                                        objectFit: "contain",
                                        filter: "drop-shadow(3px 0px black)",
                                    }}
                                />
                            </div>
                            <Typography variant="h2" mt={2} sx={{ userSelect: "none" }}>
                                <TextTransition springConfig={presets.gentle} inline>
                                    {/* {displaySide[currentIndex] ?  ((selectedPaper && !showSlider) ? reward1[currentIndex] : "?") : reward2[currentIndex]}  */}
                                    {/* {(selectedPaper && !showSlider) ? ((displaySide[currentIndex]) ? reward1[currentIndex] : reward2[currentIndex]) : (displaySide[currentIndex] ? reward1[currentIndex] : reward2[currentIndex])} */}
                                    {selectedPaper ? ((displaySide[currentIndex]) ? (oneRewarded ? reward1[currentIndex] : "-") : (!oneRewarded ? reward2[currentIndex] : "-")) : (displaySide[currentIndex] ? reward1[currentIndex] : reward2[currentIndex])}
                                </TextTransition>
                            </Typography>
                            <Box mt={2}>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    disabled={selectedPaper !== null}
                                    onClick={() => handlePaperClick("left")}
                                    startIcon={<CheckIcon />}
                                >
                                    Select
                                </Button>
                            </Box>
                        </PaperWrapper>
                    </Slide>
                    <Slide direction="left" in={newPaperAnimation} container={containerRef.current}>
                        <PaperWrapper
                            elevation={3}
                            sx={{
                                backgroundColor:
                                    selectedPaper === "right" ? "lightblue" : "white",
                            }}
                        >
                            <div style={{ userSelect: "none", display: "flex", justifyContent: "center", alignItems: "center", height: "100%", flexGrow: 1, overflow: "hidden" }}>
                                <img
                                    src={displaySide[currentIndex] ? Image2 : Image1}
                                    style={{
                                        maxWidth: "100%",
                                        maxHeight: "100%",
                                        objectFit: "contain",
                                        filter: "drop-shadow(3px 0px black)",
                                    }}
                                />
                            </div>
                            <Typography variant="h2" mt={2} sx={{ userSelect: "none" }}>
                                <TextTransition springConfig={presets.gentle} inline>
                                    {selectedPaper ? ((displaySide[currentIndex]) ? (!oneRewarded ? reward2[currentIndex] : "-") : (oneRewarded ? reward1[currentIndex] : "-")) : (displaySide[currentIndex] ? reward2[currentIndex] : reward1[currentIndex])} 
                                    {/* {(selectedPaper && !showSlider) ? "?" : (displaySide[currentIndex] ? reward2[currentIndex] : reward1[currentIndex])} */}
                                    {/* {displaySide[currentIndex] ? reward2[currentIndex] : ((selectedPaper && !showSlider) ? reward1[currentIndex] : "?")}                                     */}
                                </TextTransition>
                            </Typography>
                            <Box mt={2}>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    disabled={selectedPaper !== null}
                                    onClick={() => handlePaperClick("right")}
                                    startIcon={<CheckIcon />}
                                >
                                    Select
                                </Button>
                            </Box>
                        </PaperWrapper>
                    </Slide>
                </Stack>)}

            

                {showSlider && (
                    <Grow in={showSlider}>
                        <Box mt={2} width="100%">
                            <Typography variant="h6" textAlign="center">
                                How many points do you expect to get from this choice?
                            </Typography>
                            <Slider
                                value={sliderValue}

                                valueLabelDisplay="auto"
                                // min={1}
                                // max={99}
                                min={1}
                                max={9}
                                marks={true}
                                onChange={(_, value) => {
                                    setSliderValue(value);
                                    setSliderMoved(true);
                                }}
                            />
                            <Box mt={2} textAlign="center">
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={handleSliderConfirm}
                                    sx={{ mt: 2 }}
                                    disabled={!sliderMoved}
                                    startIcon={<CheckIcon />}
                                >
                                    Confirm
                                </Button>
                            </Box>
                        </Box>
                    </Grow>
                )}

                <Box mt={4}>
                    <Button
                        variant="contained"
                        color="secondary"
                        disabled={confirmDisabled}
                        onClick={handleConfirmClick}
                        sx={{
                            borderRadius: "12px",
                            backgroundColor: confirmDisabled ? "grey" : "secondary.main",
                        }}
                        startIcon={<ForwardIcon />}
                    >
                        Next Choice
                    </Button>
                </Box>
                {/* <Dialog
                    open={open}
                    onClose={handleClose}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description"
                >
                    <DialogTitle id="alert-dialog-title">
                        {"Win extra points!"}
                    </DialogTitle>
                    <DialogContent>
                        <DialogContentText id="alert-dialog-description">
                        {<>On this choice, if you get between <Box component="span" color="error.main">{pointRange[0][0]} and {pointRange[0][1]} points</Box>, you will win <Box component="span" color="error.main">100 extra points</Box>!</>}                        
                        </DialogContentText>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleClose} autoFocus>
                            Understood
                        </Button>
                    </DialogActions>
                </Dialog> */}

            </Box>
        </Box>
    );
}

export default Choice;