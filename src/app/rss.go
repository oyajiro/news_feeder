package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/PuerkitoBio/goquery"
	"github.com/jdkato/prose"
	"github.com/k3a/html2text"
)

func main() {
	testurl := ""
	doc := readUrl(testurl)
	div := extractText(doc)
	plain := html2text.HTML2Text(div)
	tokens := createTokens(plain)
	// fmt.Print("\n---------PLAIN-------------\n")
	// fmt.Print(plain)
	fmt.Print(tokens)
	fmt.Print("\n----------------------\n")

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

func readUrl(url string) *goquery.Document {
	resp, err := http.Get(url)
	if err != nil {
		// handle error
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		log.Fatalf("status code error: %d %s", resp.StatusCode, resp.Status)
	}
	doc, _ := goquery.NewDocumentFromReader(resp.Body)
	return doc
}

func extractText(doc *goquery.Document) string {
	return doc.Find("#story_text").First().Text()
}
