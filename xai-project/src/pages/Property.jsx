import {
    Typography,
    Box,
    Select,
    MenuItem,
    TextField,
    Button,
} from "@mui/material";

import SendIcon from "@mui/icons-material/Send";
import TipsAndUpdatesOutlinedIcon from "@mui/icons-material/TipsAndUpdatesOutlined";
import { useState } from "react";

// import { mapCity } from "../Helper";

const rowStyle = {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between", // Even spacing between items
    gap: 20, // Adjust gap to create uniform space
    alignItems: "center", // Align items vertically
};
const url = "http://127.0.0.1:5000/predict_property";

const decodeBase64 = (base64Str) => {
    const decodedBytes = atob(base64Str);
    const uint8Array = new Uint8Array(decodedBytes.length);

    for (let i = 0; i < decodedBytes.length; i++) {
        uint8Array[i] = decodedBytes.charCodeAt(i);
    }

    const decoder = new TextDecoder("utf-8", { fatal: true });
    try {
        return decoder.decode(uint8Array);
    } catch (e) {
        console.error("Failed to decode Base64 content:", e);
        return "";
    }
};

export default function Property() {
    const [formData, setFormData] = useState({
        posted_by: "0",
        under_construction: "0",
        rera: "0",
        bhk: "1",
        square_feet: "1000",
        ready_to_move: "0",
        resale: "0",
        city: "Bangalore",
        locality: "Whitefield",
    });

    const [price, setPrice] = useState(0);
    const [explaination, setExplaination] = useState("");
    const [limeHtml, setLimeHtml] = useState("");

    const fetchExplanation = async () => {
        try {
            const response = await fetch("http://localhost:5000/plot1", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            if (response.ok) {
                const data = await response.json();
                const decodedHtml = decodeBase64(data.lime_explanation_html); // Decode the base64 HTML
                console.log(decodedHtml);
                setLimeHtml(decodedHtml); // Save the decoded HTML in state
                setExplaination(data.text);
            } else {
                console.error("Failed to fetch explanation.");
            }
        } catch (error) {
            console.error("Error fetching explanation:", error);
        }
    };

    const handleChange = (event) => {
        const { name, value } = event.target;
        setFormData((prevState) => ({
            ...prevState,
            [name]: value,
        }));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        //Here, POST to Flask Server

        //Convert from strings
        formData.posted_by = +formData.posted_by;
        formData.under_construction = +formData.under_construction;
        formData.rera = +formData.rera;
        formData.bhk = +formData.bhk;
        formData.square_feet = +formData.square_feet;
        formData.ready_to_move = +formData.ready_to_move;
        formData.resale = +formData.resale;

        // console.log(formData);

        const response = await fetch(url, {
            method: "POST", // HTTP method
            headers: {
                "Content-Type": "application/json", // Specify JSON
            },
            body: JSON.stringify(formData), // Convert data to JSON string
        });

        const prediction = await response.json();

        setPrice(prediction["prediction"]);
    };

    return (
        <Box
            sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                height: "100vh",
                maxWidth: "1300px",
                margin: "0 auto",
                padding: 2, // Add padding to prevent content from being cut off
            }}
        >
            {/*Full Page Box*/}
            <Typography variant="h2" sx={{ mt: 5, mb: 5 }}>
                Property Valuation
            </Typography>
            <Typography variant="h5" sx={{ mb: 8 }}>
                We have trained a Random Forest Regressor Model to predict
                property valuations using a dataset with key features such as
                RERA approval, location, BHK (number of bedrooms, hall,
                kitchen), square footage, whether a property is ready to move,
                etc.
            </Typography>
            {/*Form Card*/}
            <Box
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 4, // Increase gap between rows
                    maxWidth: "1100px",
                    margin: "0 auto",
                    padding: 4, // Uniform padding
                    border: "1px solid #ccc",
                    borderRadius: 2,
                }}
            >
                <Box sx={rowStyle}>
                    <Typography variant="h6">Posted By:</Typography>
                    <Select
                        name="posted_by"
                        defaultValue={formData.posted_by}
                        onChange={handleChange}
                    >
                        <MenuItem value={"0"}>Owner</MenuItem>
                        <MenuItem value={"1"}>Dealer</MenuItem>
                        <MenuItem value={"2"}>Builder</MenuItem>
                    </Select>
                    <Typography variant="h6">
                        Is the property under construction?
                    </Typography>
                    <Select
                        name="under_construction"
                        defaultValue={formData.under_construction}
                        onChange={handleChange}
                    >
                        <MenuItem value={"0"}>No</MenuItem>
                        <MenuItem value={"1"}>Yes</MenuItem>
                    </Select>
                </Box>

                <Box sx={rowStyle}>
                    <Typography variant="h6">RERA Approved:</Typography>
                    <Select
                        name="rera"
                        defaultValue={formData.rera}
                        onChange={handleChange}
                    >
                        <MenuItem value={"0"}>No</MenuItem>
                        <MenuItem value={"1"}>Yes</MenuItem>
                    </Select>

                    <Typography variant="h6">Enter BHK</Typography>
                    <TextField
                        name="bhk"
                        // defaultValue={formData.bhk}
                        onChange={handleChange}
                        label="BHK"
                    />
                </Box>

                <Box sx={rowStyle}>
                    <Typography variant="h6">Enter Square Feet</Typography>
                    <TextField
                        name="square_feet"
                        // value={formData.square_feet}
                        onChange={handleChange}
                        label="Square Feet"
                    />
                    <Typography variant="h6">Is it ready to move?</Typography>
                    <Select
                        name="ready_to_move"
                        defaultValue={formData.ready_to_move}
                        onChange={handleChange}
                    >
                        <MenuItem value={"0"}>No</MenuItem>
                        <MenuItem value={"1"}>Yes</MenuItem>
                    </Select>
                </Box>

                <Box sx={rowStyle}>
                    <Typography variant="h6">
                        Is it a resale property?
                    </Typography>
                    <Select
                        name="resale"
                        defaultValue={formData.resale}
                        onChange={handleChange}
                    >
                        <MenuItem value={"0"}>No</MenuItem>
                        <MenuItem value={"1"}>Yes</MenuItem>
                    </Select>

                    <Typography variant="h6">Enter Locality</Typography>
                    <TextField
                        name="locality"
                        // value={formData.locality}
                        onChange={handleChange}
                        multiline={true}
                        label="Locality"
                    />
                </Box>

                <Box sx={{ display: "flex", flexDirection: "row", gap: 3 }}>
                    <Typography variant="h6">Enter City</Typography>
                    <TextField
                        name="city"
                        // value={formData.city}
                        onChange={handleChange}
                        label="City"
                    />
                </Box>

                <Box
                    sx={{
                        display: "flex", // Use Flexbox
                        justifyContent: "center", // Center horizontally
                    }}
                >
                    <Button
                        variant="contained"
                        type="submit"
                        color="success"
                        size="large"
                        endIcon={<SendIcon />}
                        sx={{
                            textTransform: "none",
                            fontSize: 21.5,
                            maxWidth: 250,
                            borderRadius: 2,
                            mr: 5,
                        }}
                        onClick={handleSubmit}
                    >
                        Predict Price
                    </Button>
                    <Button
                        variant="contained"
                        type="submit"
                        color="secondary"
                        size="large"
                        endIcon={<TipsAndUpdatesOutlinedIcon />}
                        sx={{
                            textTransform: "none",
                            fontSize: 21.5,
                            maxWidth: 250,
                            borderRadius: 2,
                        }}
                        onClick={fetchExplanation}
                    >
                        Explain
                    </Button>
                </Box>
            </Box>{" "}
            {/*Form Box*/}
            <Box
                sx={{
                    marginTop: 5, // Add space between the form and prediction result
                    padding: 3,
                    textAlign: "center",
                    border: "1px solid #ccc",
                    borderRadius: 2,
                    width: "100%",
                    maxWidth: "1050px",
                }}
            >
                {price === 0 ? (
                    <Typography variant="h6">No Prediction Made</Typography>
                ) : (
                    <Typography variant="h6">
                        Predicted Price By Model is Rs.{price} Lakh
                    </Typography>
                )}
            </Box>
            <Box
                sx={{
                    marginTop: 5, // Add space between the form and prediction result
                    padding: 3,
                    textAlign: "center",
                    border: "1px solid #ccc",
                    borderRadius: 2,
                    width: "100%",
                    maxWidth: "1050px",
                }}
            >
                {explaination === "" ? (
                    <Typography variant="h6">No Explaination</Typography>
                ) : (
                    <Typography variant="h6">{explaination}</Typography>
                )}
            </Box>
            <Box
                sx={{
                    marginTop: 5, // Add space between the form and prediction result
                    padding: 3,
                    textAlign: "center",
                    border: "1px solid #ccc",
                    borderRadius: 2,
                    width: "100%",
                    maxWidth: "1050px",
                }}
            >
                {limeHtml === "" ? (
                    <Typography variant="h6">No Explaination</Typography>
                ) : (
                    <iframe
                        title="LIME Explanation"
                        srcDoc={limeHtml}
                        style={{
                            width: "100%",
                            height: "500px",
                            border: "none",
                        }}
                        sandbox="allow-scripts"
                    ></iframe>
                )}
            </Box>
        </Box> //Page Box End
    );
}
