package main

import (
	"database/sql"
	"fmt"
	"log"

	// "os"
	// "os/exec"
	"strings"

	"github.com/go-sql-driver/mysql"
	"github.com/jdkato/prose"
	"github.com/magiconair/properties"
	"github.com/mmcdole/gofeed"
)

var db *sql.DB

func main() {
	p := properties.MustLoadFile("../../config.properties", properties.UTF8)
	cfg := mysql.Config{
		User:   p.MustGetString("mysql.user"),
		Passwd: p.MustGetString("mysql.password"),
		Net:    "tcp",
		Addr:   "127.0.0.1:3306",
		DBName: "db",
	}
	// Get a database handle.
	var err error
	db, err = sql.Open("mysql", cfg.FormatDSN())
	if err != nil {
		log.Fatal(err)
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
