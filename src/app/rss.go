package main

import (
	"database/sql"
	"fmt"
	"log"
	// "os"
	// "os/exec"
	"strings"

	_ "github.com/go-sql-driver/mysql"
	"github.com/jdkato/prose"
	"github.com/magiconair/properties"
	"github.com/mmcdole/gofeed"
)

func main() {
	p := properties.MustLoadFile("../../config.properties", properties.UTF8)
	dbUser := p.MustGetString("mysql.user")
	dbPassword := p.MustGetString("mysql.user")

	db, err := sql.Open("mysql", dbUser+":"+dbPassword+"@tcp(127.0.0.1:3306)/db")
	if err != nil {
		panic(err.Error())
	}

	fp := gofeed.NewParser()
	feed, _ := fp.ParseURL("https://www.reddit.com/r/science/.rss")
	// feed, _ := fp.Parse(file)

	for _, it := range feed.Items {
		if it.Title != "" {
			var moreReplaced = strings.Replace(it.Description, " <!-- more -->", "", 1)
			var row = it.Title + "\n" + moreReplaced
			insert, _ := db.Query("INSERT INTO texts(text) VALUES ('" + row + "')")
			defer insert.Close()
		}
	}
	defer db.Close()
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
