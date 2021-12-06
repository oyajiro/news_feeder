package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/jdkato/prose"
	"github.com/mmcdole/gofeed"
)

func main() {
	file, _ := os.Open("/home/user/Downloads/science.xml")
	csvFile, err := os.OpenFile("texts.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}
	csvwriter := csv.NewWriter(csvFile)
	fp := gofeed.NewParser()
	// feed, _ := fp.ParseURL("https://www.reddit.com/r/science/.rss")
	feed, _ := fp.Parse(file)

	for _, it := range feed.Items {
		if it.Title != "" {
			var moreReplaced = strings.Replace(it.Description, " <!-- more -->", "", 1)
			var row = []string{it.Title + "\n" + moreReplaced, "0"}
			_ = csvwriter.Write(row)
		}
	}
	csvwriter.Flush()
	csvFile.Close()
}

func createTokens(text string) []string {
	doc, err := prose.NewDocument(text)
	if err != nil {
		log.Fatal(err)
	}
	var textTokens []string
	for _, tok := range doc.Tokens() {
		textTokens = append(textTokens, tok.Text)
		fmt.Print("[" + tok.Text + "]\n")
	}
	return textTokens
}
