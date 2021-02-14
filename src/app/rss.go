package main

import (
	"fmt"
	"log"
	"os"

	"github.com/jdkato/prose"
	"github.com/mmcdole/gofeed"
)

func main() {
	file, _ := os.Open("/home/user/Downloads/rss")
	fp := gofeed.NewParser()
	// feed, _ := fp.ParseURL("https://www.reddit.com/r/science/.rss")
	feed, _ := fp.Parse(file)
	var result []string
	for _, it := range feed.Items {
		result = append(result, it.Title)
		tokens := createTokens(it.Content)
		fmt.Print(tokens)
		fmt.Print("\n----------------------\n")
	}

	// fmt.Printf("%v", result)
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
